using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Rigidbody))]
public class CarController : MonoBehaviour
{
    [Header("Engine / Steering")]
    public float maxEngineTorque = 1500f;     // Nm (forward)
    public float maxReverseTorque = 800f;     // Nm (reverse)
    public float maxBrakeTorque = 3000f;
    public float maxHandbrakeTorque = 6000f;
    public float maxSteeringAngle = 30f;

    [Tooltip("Optional: a transform that defines the desired center of mass (local position is used).")]
    public Transform centerOfMass;

    [System.Serializable]
    public class WheelSetup
    {
        public WheelCollider wheelCollider;
        public Transform wheelMesh;
        [Header("Behaviors")]
        public bool steering = false;
        public bool traction = false;
        public bool brake = false;
        public bool handbrake = false;
    }

    public List<WheelSetup> wheels = new List<WheelSetup>();

    [Header("Input (default)")]
    public string verticalAxis = "Vertical";   // throttle (+) / reverse (-)
    public string horizontalAxis = "Horizontal";
    public KeyCode handbrakeKey = KeyCode.Space;

    // New: allow agent control
    [Header("Agent Control")]
    public bool useAgentControl = true; // false = player input, true = agent control
    private float agentSteerInput = 0f;  // -1..+1
    private float agentThrottleInput = 0f; // -1..+1 (negative => reverse)
    private bool agentHandbrake = false;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (centerOfMass != null)
            rb.centerOfMass = centerOfMass.localPosition;

        foreach (var w in wheels)
        {
            if (w.wheelCollider == null)
                Debug.LogWarning($"CarController on '{name}' has a wheel slot with missing WheelCollider.");
            if (w.wheelMesh == null)
                Debug.LogWarning($"CarController on '{name}' has a wheel slot with missing wheelMesh (visual).");
        }
    }

    // Public API for the Agent to set controls
    public void SetControls(float steer /*-1..1*/, float throttle /*-1..1*/, bool handbrake)
    {
        agentSteerInput = Mathf.Clamp(steer, -1f, 1f);
        agentThrottleInput = Mathf.Clamp(throttle, -1f, 1f);
        agentHandbrake = handbrake;
    }

    void FixedUpdate()
    {
        float v;
        float h;
        bool handbrakeOn;

        if (useAgentControl)
        {
            v = agentThrottleInput;
            h = agentSteerInput;
            handbrakeOn = agentHandbrake;
        }
        else
        {
            v = Input.GetAxis(verticalAxis);
            h = Input.GetAxis(horizontalAxis);
            handbrakeOn = Input.GetKey(handbrakeKey);
        }

        // Count traction wheels
        int tractionCount = 0;
        foreach (var w in wheels)
            if (w.traction && w.wheelCollider != null) tractionCount++;

        // Determine total motor torque (preserve sign)
        float motorTotal;
        if (v >= 0f)
            motorTotal = v * maxEngineTorque;        // forward
        else
            motorTotal = v * maxReverseTorque;       // reverse

        float motorPerWheel = tractionCount > 0 ? motorTotal / tractionCount : 0f;

        foreach (var w in wheels)
        {
            if (w.wheelCollider == null) continue;

            // Steering
            if (w.steering)
            {
                w.wheelCollider.steerAngle = maxSteeringAngle * h;
            }

            // Motor torque
            if (w.traction)
            {
                w.wheelCollider.motorTorque = motorPerWheel;
            }
            else
            {
                w.wheelCollider.motorTorque = 0f;
            }

            // Braking
            float brakeTorque = 0f;
            if (handbrakeOn && w.handbrake)
            {
                brakeTorque = Mathf.Max(brakeTorque, maxHandbrakeTorque);
            }
            w.wheelCollider.brakeTorque = brakeTorque;

            // Visual sync
            if (w.wheelMesh != null)
            {
                Vector3 pos;
                Quaternion rot;
                w.wheelCollider.GetWorldPose(out pos, out rot);
                w.wheelMesh.position = pos;
                w.wheelMesh.rotation = rot;
            }
        }
    }
}

