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
    public int maxEpisodeSteps = 3072;

    private Vector3 startPosition;
    private Quaternion startRotation;
    private ParkingManager pmCache = null;
    private int currentStepCount = 0;
    
    // CRITICAL: Track whether goal was achieved this episode
    private bool goalAchievedThisEpisode = false;

    protected override void Awake()
    {
        base.Awake();

        if (carController == null) carController = GetComponent<CarController>();
        if (rb == null) rb = GetComponent<Rigidbody>();

        if (lidar == null)
        {
            lidar = GetComponentInChildren<LidarSensor>();
            if (lidar == null)
            {
                lidar = gameObject.AddComponent<LidarSensor>();
                lidar.rayCount = rayCount;
                lidar.rayDistance = rayDistance;
                lidar.obstacleMask = obstacleMask;
            }
        }

        if (pmCache == null)
        {
            if (transform.parent != null)
            {
                pmCache = transform.parent.GetComponentInChildren<ParkingManager>();
            }

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

        goalAchievedThisEpisode = false;

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
            sensor.AddObservation(0f);
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

        //goal achievement flag as observation
        sensor.AddObservation(goalAchievedThisEpisode ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float steer = 0f;
        float throttle = 0f;
        bool handbrake = false;

        if (actions.ContinuousActions.Length >= 1)
        steer = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        if (actions.ContinuousActions.Length >= 2)
        throttle = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        // No handbrake action anymore
        handbrake = false;

        if (carController != null)
            carController.SetControls(steer, throttle, handbrake);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cont = actionsOut.ContinuousActions;
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");

        if (cont.Length >= 1) cont[0] = Mathf.Clamp(h, -1f, 1f);
        if (cont.Length >= 2) cont[1] = Mathf.Clamp(v, -1f, 1f);
        // no cont[2]
    }

    void FixedUpdate()
    {
        RequestDecision();
        currentStepCount++;

        if (maxEpisodeSteps > 0 && currentStepCount >= maxEpisodeSteps)
        {
            Debug.Log($"[CarAgent {gameObject.name}] Episode TIMEOUT at {currentStepCount} steps (max: {maxEpisodeSteps})");
        }
    }

    Transform FindActiveGoalTransform()
    {
        if (pmCache != null && pmCache.CurrentAssignedSpot != null)
        {
            if (pmCache.CurrentAssignedSpot.spotTrigger != null)
                return pmCache.CurrentAssignedSpot.spotTrigger.transform;
            return pmCache.CurrentAssignedSpot.transform;
        }
        return null;
    }

    public void SignalGoalAchieved()
    {
        Debug.Log($"[CarAgent {gameObject.name}] SignalGoalAchieved() called - setting flag and ending episode");
        goalAchievedThisEpisode = true;
        
        EndEpisode();
    }
}

