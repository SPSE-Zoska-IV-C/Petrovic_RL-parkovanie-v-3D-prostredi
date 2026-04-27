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

    [Header("Lidar")]
    public LidarSensor lidar;
    public int rayCount = 12;
    public float rayDistance = 12f;
    public LayerMask obstacleMask;

    [Header("Goal / Episode")]
    public float maxEpisodeDistance = 50f;
    public int maxEpisodeSteps = 3072;
    public string parkingGoalTag = "ParkingGoal";

    // internal
    private Vector3 startPosition;
    private Quaternion startRotation;
    private int currentStepCount = 0;

    // +1 => success, -1 => crash, 0 => ongoing / timeout
    private float endReasonFlag = 0f;

    // cached manager for this environment
    private ParkingManager pmCache = null;

    void Awake()
    {
        if (carController == null) carController = GetComponent<CarController>();
        if (rb == null) rb = GetComponent<Rigidbody>();
        if (lidar == null) lidar = GetComponentInChildren<LidarSensor>();

        startPosition = transform.position;
        startRotation = transform.rotation;

        // try to find ParkingManager in same environment (sibling/parent)
        if (transform.parent != null)
            pmCache = transform.parent.GetComponentInChildren<ParkingManager>();
        if (pmCache == null)
            pmCache = GetComponentInChildren<ParkingManager>();
    }

    public override void Initialize()
    {
        if (carController != null)
            carController.useAgentControl = true;
    }

    public override void OnEpisodeBegin()
    {
        // reset physics & pose
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = startPosition;
        transform.rotation = startRotation;
        StartCoroutine(ResetPhysicsNextFixedUpdate());

        currentStepCount = 0;
        endReasonFlag = 0f; 

        if (pmCache == null)
            pmCache = GetComponentInChildren<ParkingManager>();

        if (pmCache != null)
            pmCache.StartRound();
    }

    private System.Collections.IEnumerator ResetPhysicsNextFixedUpdate()
    {
        rb.isKinematic = true;
        yield return new WaitForFixedUpdate();
        rb.isKinematic = false;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // lidar 
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

        // speed
        float forwardVel = Vector3.Dot(rb.linearVelocity, transform.forward);
        float normalizedSpeed = Mathf.Clamp(forwardVel / 20f, -1f, 1f);
        sensor.AddObservation(normalizedSpeed);

        // --- goal information (position + heading) ---
        Transform goalTransform = FindActiveGoalTransform();
        if (goalTransform != null)
        {
            Vector3 toGoal = goalTransform.position - transform.position;
            float dist = toGoal.magnitude;
            float normDist = Mathf.Clamp01(dist / maxEpisodeDistance);
            sensor.AddObservation(normDist);

            float signedAngle = Vector3.SignedAngle(transform.forward, toGoal.normalized, Vector3.up);
            sensor.AddObservation(Mathf.Clamp(signedAngle / 180f, -1f, 1f));

            Vector3 rel = toGoal / Mathf.Max(0.0001f, maxEpisodeDistance);
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

        sensor.AddObservation(endReasonFlag);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float steer = 0f;
        float motor = 0f;

        var cont = actions.ContinuousActions;
        if (cont.Length >= 2)
        {
            steer = Mathf.Clamp(cont[0], -1f, 1f);
            motor = Mathf.Clamp(cont[1], -1f, 1f);
        }
        else
        {
            if (cont.Length >= 1) steer = Mathf.Clamp(cont[0], -1f, 1f);
            if (cont.Length == 1) motor = 0f;
        }

        if (carController != null)
            carController.SetControls(steer, motor);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cont = actionsOut.ContinuousActions;
        if (cont.Length >= 2)
        {
            cont[0] = Input.GetAxis("Horizontal");
            cont[1] = Input.GetAxis("Vertical");
        }
        else if (cont.Length == 1)
        {
            cont[0] = Input.GetAxis("Horizontal");
        }
    }

    void FixedUpdate()
    {
        RequestDecision();
        currentStepCount++;
        if (maxEpisodeSteps > 0 && currentStepCount >= maxEpisodeSteps)
        {
            EndEpisode();
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

    public void SignalEpisodeEnd(float flag)
    {
        // expected: +1 => success, -1 => crash, 0 => (unused here)
        endReasonFlag = Mathf.Clamp(flag, -1f, 1f);

        Debug.Log($"[CarAgent] SignalEpisodeEnd flag={endReasonFlag} step={currentStepCount}");

        EndEpisode();
    }
}
