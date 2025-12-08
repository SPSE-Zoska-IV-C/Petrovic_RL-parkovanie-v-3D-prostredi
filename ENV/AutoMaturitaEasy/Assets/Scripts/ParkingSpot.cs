using UnityEngine;

[DisallowMultipleComponent]
public class ParkingSpot : MonoBehaviour
{
    [Header("Assign in Inspector")]
    public Collider spotTrigger;     // the Box Collider (set IsTrigger = true)

    [HideInInspector] public bool isGoal = false;      // becomes true AFTER successful stay
    [HideInInspector] public bool isAssigned = false;  // set by ParkingManager when this spot is the chosen target

    [Tooltip("Seconds the player must remain fully inside the trigger before completing the goal")]
    public float requiredStaySeconds = 2f; // CHANGED: reduced for easier learning

    [Tooltip("Enable to get debug logs (throttle applied). Disable in release to avoid allocations/log spam).")]
    public bool debugLogs = true;

    [Header("Goal / Pause Options")]
    [Tooltip("If true, the game will be paused when this spot becomes the goal (Time.timeScale = 0).")]
    public bool pauseOnGoal = false;

    [Tooltip("If true and running inside the Editor, stop Play Mode when the goal is reached.")]
    public bool stopEditorPlayMode = false;

    // runtime tracking state (no allocations inside physics callbacks)
    private GameObject trackingRoot = null;
    private Collider[] trackingColliders = null;
    private float accumulatedInsideTime = 0f;
    private float lastProgressLogTime = 0f;
    private const float PROGRESS_LOG_INTERVAL = 1f;

    // saved time settings for resume
    private float savedTimeScale = 1f;
    private float savedFixedDeltaTime = 0.02f;

    // -------- lifecycle --------
    public void ResetSpot()
    {
        isGoal = false;
        isAssigned = false;
        StopTracking();

        if (spotTrigger != null)
            spotTrigger.enabled = false;

        // restore time/audio if we paused the game previously
        ResumeGame();

        if (debugLogs) Debug.Log($"[PS] ResetSpot: {gameObject.name}");
    }

    // -------- physics callbacks (cheap, allocation-free) --------

    private void OnTriggerEnter(Collider other)
    {
        if (!isAssigned) return;
        if (!IsPlayerCandidate(other)) return;

        GameObject root = ResolveRoot(other);

        if (trackingRoot == null)
        {
            StartTracking(root);
            if (debugLogs) Debug.Log($"[PS] Candidate entered (not yet fully inside): {root.name} on {gameObject.name}");
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
                Debug.Log($"[PS] Within bounds: {trackingRoot.name} started waiting for {requiredStaySeconds} s on {gameObject.name}");
                lastProgressLogTime = Time.time - PROGRESS_LOG_INTERVAL;
            }

            accumulatedInsideTime += Time.fixedDeltaTime;

            if (debugLogs && Time.time - lastProgressLogTime >= PROGRESS_LOG_INTERVAL)
            {
                float remaining = Mathf.Max(0f, requiredStaySeconds - accumulatedInsideTime);
                Debug.Log($"[PS] Still within bounds ({trackingRoot.name}) on {gameObject.name}. Waiting... {remaining:F1}s left");
                lastProgressLogTime = Time.time;
            }

            if (accumulatedInsideTime >= requiredStaySeconds)
            {
                CompleteGoal(trackingRoot);
            }
        }
        else
        {
            if (accumulatedInsideTime > 0f && debugLogs)
                Debug.Log($"[PS] Not fully inside anymore: cancelling wait for {gameObject.name}");

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
            if (debugLogs) Debug.Log($"[PS] Candidate left trigger: cancelling wait for {gameObject.name}");
            StopTracking();
        }
    }

    // -------- helpers (no allocations) --------

    private GameObject ResolveRoot(Collider other)
    {
        return (other.attachedRigidbody != null) ? other.attachedRigidbody.gameObject : other.transform.root.gameObject;
    }

    private bool IsPlayerCandidate(Collider other)
    {
        GameObject root = ResolveRoot(other);
        if (root == null) return false;

        // Check for Player tag or Rigidbody (the car agent)
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
        if (spotTrigger == null || trackingColliders == null || trackingColliders.Length == 0)
            return false;

        Bounds triggerBounds = spotTrigger.bounds;

        for (int i = 0; i < trackingColliders.Length; ++i)
        {
            var c = trackingColliders[i];
            if (c == null) continue;
            if (c == spotTrigger) continue;

            Bounds b = c.bounds;
            if (!triggerBounds.Contains(b.min) || !triggerBounds.Contains(b.max))
                return false;
        }

        return true;
    }

    private void CompleteGoal(GameObject byObject)
    {
        if (!isAssigned) return; // extra safety

        isGoal = true;
        isAssigned = false;

        if (spotTrigger != null)
            spotTrigger.enabled = false;

        if (debugLogs) Debug.Log($"[PS] >>> GOAL ACHIEVED: {gameObject.name} by {byObject.name} <<< (isGoal set true)");

        StopTracking();

        var agent = byObject.GetComponentInParent<CarAgent>();
        if (agent != null)
        {
            var rb = agent.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.isKinematic = false; // ensure physics is on for the next episode
            }

            // Signal episode end once: ParkingSpot is the owner.
            agent.EndEpisode();
        }
    }

    // Minimal pause/resume helpers
    private void PauseGame()
    {
        // Save current time settings
        savedTimeScale = Time.timeScale;
        savedFixedDeltaTime = Time.fixedDeltaTime;

        // Pause gameplay
        Time.timeScale = 0f;
        Time.fixedDeltaTime = 0f;
        AudioListener.pause = true;

#if UNITY_EDITOR
        if (stopEditorPlayMode)
        {
            UnityEditor.EditorApplication.isPlaying = false;
        }
#endif
    }

    private void ResumeGame()
    {
        // Restore time/audio
        Time.timeScale = savedTimeScale;
        Time.fixedDeltaTime = savedFixedDeltaTime;
        AudioListener.pause = false;
    }
}