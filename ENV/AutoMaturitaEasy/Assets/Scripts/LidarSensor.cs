
using UnityEngine;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

[DisallowMultipleComponent]
public class LidarSensor : MonoBehaviour
{
    [Header("Lidar settings")]
    [Tooltip("Number of rays (8-12 recommended).")]
    public int rayCount = 12;
    [Tooltip("Max ray distance in meters.")]
    public float rayDistance = 12f;
    [Tooltip("Local offset for the ray origin (e.g. raise above ground).")]
    public Vector3 originOffset = new Vector3(0f, 0.25f, 0f);
    [Tooltip("Layer mask for objects the rays should detect.")]
    public LayerMask obstacleMask = ~0; // default: everything
    [Tooltip("If true, rays will detect trigger colliders as well.")]
    public QueryTriggerInteraction triggerInteraction = QueryTriggerInteraction.Collide;
    [Tooltip("Draw debug rays when the game is running.")]
    public bool debugDraw = false;
    [Tooltip("Start angle offset (degrees). 0 = forward, 180 = backward etc.")]
    public float startAngle = 0f;

    public void AddObservations(VectorSensor sensor)
    {
        float[] arr = ComputeObservationsArray();
        for (int i = 0; i < arr.Length; ++i)
            sensor.AddObservation(arr[i]);
    }

    public float[] GetObservationsArray()
    {
        return ComputeObservationsArray();
    }

    private float[] ComputeObservationsArray()
    {
        int useRayCount = Mathf.Max(1, rayCount);
        float[] arr = new float[useRayCount];
        Vector3 origin = transform.position + transform.TransformVector(originOffset);
        float angleStep = 360f / useRayCount;

        for (int i = 0; i < useRayCount; ++i)
        {
            float angle = startAngle + i * angleStep;
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * transform.forward;

            RaycastHit hit;
            bool gotHit = Physics.Raycast(origin, dir, out hit, rayDistance, obstacleMask, triggerInteraction);

            float obs = 0f;
            if (gotHit)
            {
                obs = Mathf.Clamp01(hit.distance / rayDistance);
            }
            else
            {
                obs = 1f;
            }

            arr[i] = obs;

            if (debugDraw)
            {
                Color c = gotHit ? Color.red : Color.green;
                Debug.DrawRay(origin, dir * (gotHit ? hit.distance : rayDistance), c);
            }
        }

        return arr;
    }
}