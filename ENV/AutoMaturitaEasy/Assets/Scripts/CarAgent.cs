
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

[RequireComponent(typeof(Rigidbody))]
public class CarAgent : Agent
{
    [Header("Refs")]
    public CarController carController;
    public Rigidbody rb;

    [Header("Lidar (raycasts)")]
    [Tooltip("Optional: If assigned, this LidarSensor component will produce the ray observations.")]
    public LidarSensor lidar;
    [Tooltip("Fallback: number of rays if Lidar component is not assigned")]
    public int rayCount = 12;
    public float rayDistance = 12f;
    public LayerMask obstacleMask;

    [Header("Goal / Episode")]
    [Tooltip("Tag name of the parking spot trigger object that counts as the goal.")]
    public string parkingGoalTag = "ParkingGoal";
    [Tooltip("Tag name for objects that should cause crash penalty.")]
    public string forbiddenTag = "Forbidden";
    public float goalHoldTime = 5f;
    public float maxEpisodeDistance = 50f;

    [Header("Episode Limits")]
    [Tooltip("Maximum steps before episode ends (0 = unlimited). Recommended: 3072 for easy environment")]
    public int maxEpisodeSteps = 3072; // UPDATED: Increased from 2048

    // internal
    private float insideGoalTimer = 0f;
    private bool isInsideGoal = false;
    private Vector3 startPosition;
    private Quaternion startRotation;
    private ParkingManager pmCache = null;
    private int currentStepCount = 0;

    protected override void Awake()
    {
        base.Awake();
        
        // ALWAYS auto-find components (safe for cloning)
        if (carController == null) carController = GetComponent<CarController>();
        if (rb == null) rb = GetComponent<Rigidbody>();
        
        // Auto-find lidar if not assigned
        if (lidar == null) lidar = GetComponentInChildren<LidarSensor>();
        
        // CRITICAL FIX: Find ParkingManager in THIS environment only
        if (pmCache == null)
        {
            // First try: look in parent (if agent is under ParkingEnvironment)
            if (transform.parent != null)
            {
                pmCache = transform.parent.GetComponentInChildren<ParkingManager>();
            }
            
            // Fallback: look in children (if ParkingManager is somehow under this agent)
            if (pmCache == null)
            {
                pmCache = GetComponentInChildren<ParkingManager>();
            }
            
            if (pmCache != null)
            {
                Debug.Log($"[CarAgent {gameObject.name}] Found ParkingManager: {pmCache.gameObject.name}");
            }
            else
            {
                Debug.LogWarning($"[CarAgent {gameObject.name}] No ParkingManager found in this environment!");
            }
        }
        
        startPosition = transform.position;
        startRotation = transform.rotation;
    }

    public override void Initialize()
    {
        if (carController != null)
            carController.useAgentControl = true;

        if (Academy.Instance != null)
        {
            Debug.Log($"[CarAgent {gameObject.name}] Academy is initialized.");
        }
        else
        {
            Debug.Log($"[CarAgent {gameObject.name}] Academy is not initialized.");
        }
    }

    private System.Collections.IEnumerator SafeResetPhysics()
    {
        rb.isKinematic = true;
        transform.position = startPosition;
        transform.rotation = startRotation;
        yield return new WaitForFixedUpdate();
        rb.isKinematic = false;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }

    public override void OnEpisodeBegin()
    {
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.position = startPosition;
        transform.rotation = startRotation;

        StartCoroutine(SafeResetPhysics());
        insideGoalTimer = 0f;
        isInsideGoal = false;

        // CRITICAL FIX: Use cached ParkingManager (found in Awake)
        // This ensures each agent uses its OWN environment's manager
        if (pmCache != null)
        {
            pmCache.StartRound();
        }
        else
        {
            Debug.LogWarning($"[CarAgent {gameObject.name}] No ParkingManager cached! Episode may not work correctly.");
        }

        currentStepCount = 0;
        Debug.Log($"[CarAgent {gameObject.name}] Episode started at {transform.position}");
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1) Lidar rays
        if (lidar != null)
        {
            lidar.AddObservations(sensor);
        }
        else
        {
            Vector3 origin = transform.position + transform.TransformVector(new Vector3(0f, 0.25f, 0f));
            int useRayCount = Mathf.Max(1, rayCount);
            float angleStep = 360f / useRayCount;

            for (int i = 0; i < useRayCount; ++i)
            {
                float angle = i * angleStep;
                Vector3 dir = Quaternion.Euler(0f, angle, 0f) * transform.forward;
                RaycastHit hit;
                if (Physics.Raycast(origin, dir, out hit, rayDistance, obstacleMask, QueryTriggerInteraction.Collide))
                {
                    float n = 1f - Mathf.Clamp01(hit.distance / rayDistance);
                    sensor.AddObservation(n);
                }
                else
                {
                    sensor.AddObservation(0f);
                }
            }
        }

        // 2) Speed
        float forwardVel = Vector3.Dot(rb.linearVelocity, transform.forward);
        float normalizedSpeed = Mathf.Clamp(forwardVel / 20f, -1f, 1f);
        sensor.AddObservation(normalizedSpeed);

        // 3) Goal information
        Transform goalTransform = FindActiveGoalTransform();
        if (goalTransform != null)
        {
            Vector3 toGoalWorld = goalTransform.position - transform.position;
            float dist = toGoalWorld.magnitude;
            float normDist = Mathf.Clamp01(dist / maxEpisodeDistance);
            sensor.AddObservation(normDist);

            float signedAngle = Vector3.SignedAngle(transform.forward, toGoalWorld.normalized, Vector3.up);
            sensor.AddObservation(Mathf.Clamp(signedAngle / 180f, -1f, 1f));

            Vector3 rel = toGoalWorld / Mathf.Max(0.0001f, maxEpisodeDistance);
            sensor.AddObservation(Mathf.Clamp(rel.x, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(rel.y, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(rel.z, -1f, 1f));
        }
        else
        {
            sensor.AddObservation(1f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Continuous actions expected in order: [steer, throttle, handbrake]
        // steer: -1 .. +1, throttle: -1 .. +1, handbrake: 0 .. 1
        float steer = 0f;
        float throttle = 0f;
        bool handbrake = false;

        if (actions.ContinuousActions.Length >= 1)
            steer = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        if (actions.ContinuousActions.Length >= 2)
            throttle = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        if (actions.ContinuousActions.Length >= 3)
        {
            // Interpret handbrake as continuous (0..1), threshold to boolean.
            float hb = Mathf.Clamp01(actions.ContinuousActions[2]);
            handbrake = hb >= 0.5f;
        }

        if (carController != null)
            carController.SetControls(steer, throttle, handbrake);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Output continuous actions so you can test with keyboard.
        var cont = actionsOut.ContinuousActions;
        // steer: A/D or arrow keys
        float h = Input.GetAxis("Horizontal");  // -1..1
        // throttle: W/S or vertical axis
        float v = Input.GetAxis("Vertical");    // -1..1
        // handbrake: space -> 1, else 0
        float hb = Input.GetKey(KeyCode.Space) ? 1f : 0f;

        if (cont.Length >= 1) cont[0] = Mathf.Clamp(h, -1f, 1f);
        if (cont.Length >= 2) cont[1] = Mathf.Clamp(v, -1f, 1f);
        if (cont.Length >= 3) cont[2] = Mathf.Clamp01(hb);
    }

    void OnTriggerEnter(Collider other)
    {
        if (other == null) return;

        var spot = other.GetComponentInParent<ParkingSpot>();
        if (spot != null)
        {
            // Use cached pmCache instead of FindFirstObjectByType
            if (pmCache != null && pmCache.CurrentAssignedSpot == spot)
            {
                isInsideGoal = true;
                insideGoalTimer = 0f;
                return;
            }

            if (spot.spotTrigger != null && spot.spotTrigger.enabled)
            {
                isInsideGoal = true;
                insideGoalTimer = 0f;
                return;
            }
        }

        if (!string.IsNullOrEmpty(parkingGoalTag))
        {
            try
            {
                if (other.CompareTag(parkingGoalTag))
                {
                    isInsideGoal = true;
                    insideGoalTimer = 0f;
                    return;
                }
            }
            catch (UnityException)
            {
                Debug.LogWarning($"CarAgent.OnTriggerEnter: tag '{parkingGoalTag}' not defined.");
            }
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other == null) return;

        var spot = other.GetComponentInParent<ParkingSpot>();
        if (spot != null)
        {
            if (spot.isGoal || (spot.spotTrigger != null && spot.spotTrigger.enabled))
            {
                isInsideGoal = false;
                insideGoalTimer = 0f;
                return;
            }
        }

        if (!string.IsNullOrEmpty(parkingGoalTag))
        {
            try
            {
                if (other.CompareTag(parkingGoalTag))
                {
                    isInsideGoal = false;
                    insideGoalTimer = 0f;
                    return;
                }
            }
            catch (UnityException)
            {
                Debug.LogWarning($"CarAgent.OnTriggerExit: tag '{parkingGoalTag}' not defined.");
            }
        }
    }

    void Update()
    {
        if (isInsideGoal)
        {
            insideGoalTimer += Time.deltaTime;
        }
        else
        {
            if (insideGoalTimer != 0f)
                insideGoalTimer = 0f;
        }
    }

    void FixedUpdate()
    {
        RequestDecision();
        currentStepCount++;
    
        if (maxEpisodeSteps > 0 && currentStepCount >= maxEpisodeSteps)
        {
            Debug.Log($"[CarAgent {gameObject.name}] Episode truncated at {currentStepCount} steps (max: {maxEpisodeSteps})");
            EndEpisode();
        }
    }

    Transform FindActiveGoalTransform()
    {
        // Use cached ParkingManager instead of finding globally
        if (pmCache != null && pmCache.CurrentAssignedSpot != null)
        {
            if (pmCache.CurrentAssignedSpot.spotTrigger != null)
                return pmCache.CurrentAssignedSpot.spotTrigger.transform;
            return pmCache.CurrentAssignedSpot.transform;
        }
        return null;
    }
}