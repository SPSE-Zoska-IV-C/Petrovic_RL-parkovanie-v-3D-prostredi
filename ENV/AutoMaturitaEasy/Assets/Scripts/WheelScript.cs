using UnityEngine;

[RequireComponent(typeof(WheelCollider))]
public class WheelScript : MonoBehaviour
{
    [Tooltip("The visual model (mesh) for this wheel. This is what will be positioned/rotated to match the WheelCollider.")]
    public Transform wheelMesh;

    private WheelCollider wc;

    void Awake()
    {
        wc = GetComponent<WheelCollider>();
        if (wheelMesh == null)
            Debug.LogWarning($"WheelScript on '{name}' has no wheelMesh assigned.");
    }

    void Update()
    {
        if (wheelMesh == null || wc == null) return;
        Vector3 pos;
        Quaternion rot;
        wc.GetWorldPose(out pos, out rot);
        wheelMesh.position = pos;
        wheelMesh.rotation = rot;
    }

    public void ApplySteer(float angle) => wc.steerAngle = angle;
    public void ApplyMotorTorque(float torque) => wc.motorTorque = torque;
    public void ApplyBrakeTorque(float torque) => wc.brakeTorque = torque;
    public WheelCollider Collider => wc;
}
