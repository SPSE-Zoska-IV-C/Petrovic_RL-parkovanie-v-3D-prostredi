using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Rigidbody))]
public class CarController : MonoBehaviour
{
    [Header("Engine / Steering")]
    public float maxEngineTorque = 1500f;     
    public float maxReverseTorque = 800f;     
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
    public string verticalAxis = "Vertical";  
    public string horizontalAxis = "Horizontal";
    public KeyCode handbrakeKey = KeyCode.Space;

    // New: allow agent control
    [Header("Agent Control")]
    public bool useAgentControl = true; 
    private float agentSteerInput = 0f;  
    private float agentThrottleInput = 0f; 
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

    public void SetControls(float steer /*-1..1*/, float throttle /*-1..1*/)
    {
        agentSteerInput = Mathf.Clamp(steer, -1f, 1f);
        agentThrottleInput = Mathf.Clamp(throttle, -1f, 1f);
        agentHandbrake = false;
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

        int tractionCount = 0;
        foreach (var w in wheels)
            if (w.traction && w.wheelCollider != null) tractionCount++;

        float motorTotal;
        if (v >= 0f)
            motorTotal = v * maxEngineTorque;        
        else
            motorTotal = v * maxReverseTorque;       

        float motorPerWheel = tractionCount > 0 ? motorTotal / tractionCount : 0f;

        float forwardVel = Vector3.Dot(rb.linearVelocity, transform.forward);

        foreach (var w in wheels)
        {
            if (w.wheelCollider == null) continue;

            if (w.steering)
            {
                w.wheelCollider.steerAngle = maxSteeringAngle * h;
            }

            float brakeTorque = 0f;
            if (handbrakeOn && w.handbrake)
            {
                brakeTorque = Mathf.Max(brakeTorque, maxHandbrakeTorque);
            }

            if (w.brake)
            {
                if (Mathf.Abs(v) > 0.01f && Mathf.Sign(forwardVel) != 0f && Mathf.Sign(v) != Mathf.Sign(forwardVel))
                {
                    brakeTorque = Mathf.Max(brakeTorque, Mathf.Abs(v) * maxBrakeTorque);
                }
            }

            if (w.traction)
            {
                if (brakeTorque > 0.0f)
                    w.wheelCollider.motorTorque = 0f;
                else
                    w.wheelCollider.motorTorque = motorPerWheel;
            }
            else
            {
                w.wheelCollider.motorTorque = 0f;
            }

            w.wheelCollider.brakeTorque = brakeTorque;

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