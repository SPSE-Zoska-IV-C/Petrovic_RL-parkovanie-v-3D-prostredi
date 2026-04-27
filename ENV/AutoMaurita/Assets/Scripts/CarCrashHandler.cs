using UnityEngine;
using System.Collections.Generic;

[DisallowMultipleComponent]
public class CarCrashHandler : MonoBehaviour
{
    public List<GameObject> forbiddenObjects = new List<GameObject>();
    public CarAgent agentReference = null;
    public bool delayEndEpisodeOneFrame = false;

    private bool hasCrashed = false;

    private void Awake()
    {
        if (agentReference == null)
        {
            agentReference = GetComponent<CarAgent>();
            if (agentReference == null) agentReference = GetComponentInParent<CarAgent>();
            if (agentReference == null) agentReference = GetComponentInChildren<CarAgent>();
        }
    }

    private void OnCollisionEnter(Collision collision) => HandleCollision(collision.gameObject);
    private void OnTriggerEnter(Collider other) => HandleCollision(other.gameObject);

    private void HandleCollision(GameObject other)
    {
        if (hasCrashed || other == null) return;
        if (!IsForbidden(other)) return;

        hasCrashed = true;
        Debug.Log($"[CarCrashHandler] Crashed into: {other.name}");

        if (agentReference == null)
        {
            Debug.LogWarning("[CarCrashHandler] No agent reference!");
            hasCrashed = false;
            return;
        }

        var rb = agentReference.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        try
        {
            agentReference.SignalEpisodeEnd(-1f);
            Debug.Log("[CarCrashHandler] Signaled crash termination immediately.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[CarCrashHandler] Exception while signaling crash: {ex}");
        }

        StartCoroutine(ResetHasCrashedNextFrame());
    }

    private System.Collections.IEnumerator ResetHasCrashedNextFrame()
    {
        yield return new WaitForEndOfFrame();
        hasCrashed = false;
    }

    private bool IsForbidden(GameObject other)
    {
        for (int i = 0; i < forbiddenObjects.Count; ++i)
        {
            var f = forbiddenObjects[i];
            if (f == null) continue;

            if (other == f) return true;
            if (other.transform.IsChildOf(f.transform)) return true;
            if (f.transform.IsChildOf(other.transform)) return true;
        }
        return false;
    }
}
