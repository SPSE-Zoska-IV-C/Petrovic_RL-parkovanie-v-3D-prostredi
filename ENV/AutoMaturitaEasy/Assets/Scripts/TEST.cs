using UnityEngine;
using System.Collections;

public class CarControllerDiagnosticsFixed : MonoBehaviour
{
    public MonoBehaviour carController; // drag CarController component here
    public float torqueThreshold = 200f;
    public float steerThreshold = 25f;
    public float logOnEnableDelay = 0.05f;

    void OnEnable()
    {
        if (carController == null) {
            Debug.LogWarning("[Diag] assign CarController component to diagnostics");
            return;
        }
        StartCoroutine(WaitAndCheck());
    }

    IEnumerator WaitAndCheck()
    {
        float time = 0f;
        while (time < logOnEnableDelay)
        {
            time += Time.fixedDeltaTime;
            yield return new WaitForFixedUpdate();
        }

        var wheels = GetComponentsInChildren<WheelCollider>(true);
        Debug.Log($"[Diag] Found {wheels.Length} WheelColliders");
        foreach (var w in wheels)
        {
            Debug.Log($"[Diag] {w.name} localPos={w.transform.localPosition} radius={w.radius} suspensionDist={w.suspensionDistance} forceAppPointDistance={w.forceAppPointDistance} isGrounded={w.isGrounded} motorTorque={w.motorTorque} steerAngle={w.steerAngle}");
            if (Mathf.Abs(w.motorTorque) > torqueThreshold) Debug.LogError($"[Diag] HIGH motorTorque on {w.name}: {w.motorTorque}");
            if (Mathf.Abs(w.steerAngle) > steerThreshold) Debug.LogError($"[Diag] HIGH steerAngle on {w.name}: {w.steerAngle}");
        }

        var rb = GetComponentInChildren<Rigidbody>();
        if (rb != null) {
            Debug.Log($"[Diag] Rigidbody: name={rb.name} mass={rb.mass} CoM(local)={rb.centerOfMass} CoM(world)={rb.transform.TransformPoint(rb.centerOfMass)} velocity={rb.linearVelocity} angularVel={rb.angularVelocity}");
        } else {
            Debug.LogWarning("[Diag] No Rigidbody found in children!");
        }
    }
}
