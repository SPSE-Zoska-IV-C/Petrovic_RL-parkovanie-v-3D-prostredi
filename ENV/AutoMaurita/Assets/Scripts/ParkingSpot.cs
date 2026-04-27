using UnityEngine;

[DisallowMultipleComponent]
public class ParkingSpot : MonoBehaviour
{
    public GameObject parkedCar;
    public Collider spotTrigger;

    [HideInInspector] public bool isGoal = false;
    [HideInInspector] public bool isAssigned = false;

    public float requiredStaySeconds = 4f;
    public bool debugLogs = true;
    public bool pauseOnGoal = false;
    public bool stopEditorPlayMode = false;

    // runtime
    private GameObject trackingRoot = null;
    private Collider[] trackingColliders = null;
    private float accumulatedInsideTime = 0f;
    private float lastProgressLogTime = 0f;
    private const float PROGRESS_LOG_INTERVAL = 1f;

    private float savedTimeScale = 1f;
    private float savedFixedDeltaTime = 0.02f;

    public void ResetSpot()
    {
        isGoal = false;
        isAssigned = false;
        StopTracking();

        if (parkedCar != null)
        {
            parkedCar.SetActive(true);
            ResetCarComponents();
        }

        if (spotTrigger != null)
            spotTrigger.enabled = false;

        ResumeGame();

        if (debugLogs) Debug.Log($"[PS] ResetSpot: {gameObject.name}");
    }

    private void ResetCarComponents()
    {
        if (parkedCar == null) return;
        foreach (var r in parkedCar.GetComponentsInChildren<Renderer>(true)) r.enabled = true;
        foreach (var c in parkedCar.GetComponentsInChildren<Collider>(true)) { if (c == spotTrigger) continue; c.enabled = true; }
        foreach (var rb in parkedCar.GetComponentsInChildren<Rigidbody>(true)) rb.isKinematic = false;
    }

    public void FreeSpot()
    {
        if (parkedCar == null) return;
        parkedCar.SetActive(false);
        if (debugLogs) Debug.Log($"[PS] FreeSpot: {gameObject.name}");
    }

    public void FreeSpot_DebugForceHide()
    {
        if (parkedCar == null)
        {
            if (debugLogs) Debug.LogWarning($"[PS] FreeSpot_Debug: parkedCar is null on {gameObject.name}");
            return;
        }
        parkedCar.SetActive(false);
        if (debugLogs) Debug.Log($"[PS] FreeSpot_Debug: SUCCESS - {parkedCar.name} hidden on {gameObject.name}");
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!isAssigned) return;
        if (!IsPlayerCandidate(other)) return;

        GameObject root = ResolveRoot(other);
        if (trackingRoot == null)
        {
            StartTracking(root);
            if (debugLogs) Debug.Log($"[PS] Candidate entered: {root.name} on {gameObject.name}");
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (!isAssigned) return;
        if (trackingRoot == null) return;

        GameObject root = ResolveRoot(other);
        if (root != trackingRoot) return;

        if (AreAllCollidersFullyInsideTrigger())
        {
            if (accumulatedInsideTime <= 0f && debugLogs)
            {
                Debug.Log($"[PS] {trackingRoot.name} started waiting for {requiredStaySeconds} s on {gameObject.name}");
                lastProgressLogTime = Time.time - PROGRESS_LOG_INTERVAL;
            }

            accumulatedInsideTime += Time.fixedDeltaTime;

            if (debugLogs && Time.time - lastProgressLogTime >= PROGRESS_LOG_INTERVAL)
            {
                float remaining = Mathf.Max(0f, requiredStaySeconds - accumulatedInsideTime);
                Debug.Log($"[PS] Waiting... {remaining:F1}s left on {gameObject.name}");
                lastProgressLogTime = Time.time;
            }

            if (accumulatedInsideTime >= requiredStaySeconds)
            {
                CompleteGoal(trackingRoot);
            }
        }
        else
        {
            if (accumulatedInsideTime > 0f && debugLogs) Debug.Log($"[PS] Not fully inside: cancel wait for {gameObject.name}");
            ResetAccumulation();
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (!isAssigned) return;
        if (trackingRoot == null) return;

        GameObject root = ResolveRoot(other);
        if (root == trackingRoot)
        {
            if (debugLogs) Debug.Log($"[PS] Candidate left trigger for {gameObject.name}");
            StopTracking();
        }
    }

    private GameObject ResolveRoot(Collider other)
    {
        return (other.attachedRigidbody != null) ? other.attachedRigidbody.gameObject : other.transform.root.gameObject;
    }

    private bool IsPlayerCandidate(Collider other)
    {
        GameObject root = ResolveRoot(other);
        if (root == null) return false;
        if (parkedCar != null && root == parkedCar) return true;
        if (root.CompareTag("Player")) return true;
        if (other.attachedRigidbody != null) return true;
        return false;
    }

    private void StartTracking(GameObject root)
    {
        trackingRoot = root;
        trackingColliders = trackingRoot.GetComponentsInChildren<Collider>(true);
        accumulatedInsideTime = 0f;
        lastProgressLogTime = 0f;
    }

    private void StopTracking()
    {
        trackingRoot = null;
        trackingColliders = null;
        ResetAccumulation();
    }

    private void ResetAccumulation()
    {
        accumulatedInsideTime = 0f;
        lastProgressLogTime = 0f;
    }

    private bool AreAllCollidersFullyInsideTrigger()
    {
        if (spotTrigger == null || trackingColliders == null || trackingColliders.Length == 0) return false;
        Bounds triggerBounds = spotTrigger.bounds;
        for (int i = 0; i < trackingColliders.Length; ++i)
        {
            var c = trackingColliders[i];
            if (c == null) continue;
            if (c == spotTrigger) continue;
            Bounds b = c.bounds;
            if (!triggerBounds.Contains(b.min) || !triggerBounds.Contains(b.max)) return false;
        }
        return true;
    }

    private void CompleteGoal(GameObject byObject)
    {
        if (!isAssigned) return;
        isGoal = true;
        isAssigned = false;
        if (spotTrigger != null) spotTrigger.enabled = false;

        if (debugLogs) Debug.Log($"[PS] GOAL ACHIEVED: {gameObject.name} by {byObject.name}");

        StopTracking();

        var agent = byObject.GetComponentInParent<CarAgent>();
        if (agent != null)
        {
            var rb = agent.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.isKinematic = false;
            }

            // MATCH easy version: agent owns the semantics. Signal with +1.
            Debug.Log($"[PS] Signaling goal (+1) to agent {agent.name}");
            agent.SignalEpisodeEnd(+1f);
        }
    }

    private void PauseGame()
    {
        savedTimeScale = Time.timeScale;
        savedFixedDeltaTime = Time.fixedDeltaTime;
        Time.timeScale = 0f;
        Time.fixedDeltaTime = 0f;
        AudioListener.pause = true;
#if UNITY_EDITOR
        if (stopEditorPlayMode)
            UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    private void ResumeGame()
    {
        Time.timeScale = savedTimeScale;
        Time.fixedDeltaTime = savedFixedDeltaTime;
        AudioListener.pause = false;
    }
}
