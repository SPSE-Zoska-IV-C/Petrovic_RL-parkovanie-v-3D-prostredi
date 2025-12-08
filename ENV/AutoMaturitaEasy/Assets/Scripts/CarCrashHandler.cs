// using UnityEngine;
// using System.Collections.Generic;

// [DisallowMultipleComponent]
// public class CarCrashHandler : MonoBehaviour
// {
//     [Tooltip("Assign objects (with BoxCollider/Collider) that will cause the episode to end when touched.")]
//     public List<GameObject> forbiddenObjects = new List<GameObject>();

//     [Tooltip("If assigned, this agent will be signalled on crash. If null, the first CarAgent found will be used.")]
//     public CarAgent agentReference = null;

//     [Tooltip("If true, EndEpisode will be delayed until the next frame to avoid sensor-reset race conditions.")]
//     public bool delayEndEpisodeOneFrame = true;

//     // Prevent repeated crash handling within the same physics step
//     private bool hasCrashed = false;

//     // Handle both physics collisions and trigger zones
//     private void OnCollisionEnter(Collision collision) => HandleCollision(collision.gameObject);
//     private void OnTriggerEnter(Collider other) => HandleCollision(other.gameObject);

//     private void HandleCollision(GameObject other)
//     {
//         if (hasCrashed || other == null) return;
//         if (!IsForbidden(other)) return;

//         hasCrashed = true;
//         Debug.Log("[CarCrashHandler] Car crashed");
//         Debug.Log($"[CarCrashHandler] Car crashed into: {other.name}");

//         CarAgent agent = agentReference;
//         if (agent == null)
//             agent = UnityEngine.Object.FindFirstObjectByType<CarAgent>();

//         if (agent == null)
//         {
//             Debug.LogWarning("[CarCrashHandler] No CarAgent found to EndEpisode(). Crash handled but no agent to notify.");
//             // nothing to notify — allow future crashes again
//             hasCrashed = false;
//             return;
//         }

//         // diagnostic: does agent look correctly configured?
//         var bp = agent.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
//         Debug.Log($"[CarCrashHandler] Ending episode on agent '{agent.name}'. BehaviorParameters present: {(bp != null)}");

//         if (delayEndEpisodeOneFrame)
//         {
//             // deferred path: do the safe EndEpisode on the next frame
//             StartCoroutine(EndEpisodeNextFrame(agent));
//         }
//         else
//         {
//             // immediate path: clear physics, then end episode now
//             var rb = agent.GetComponent<Rigidbody>();
//             if (rb != null)
//             {
//                 rb.linearVelocity = Vector3.zero;
//                 rb.angularVelocity = Vector3.zero;
//             }

//             try
//             {
//                 agent.EndEpisode();
//                 Debug.Log("[CarCrashHandler] EndEpisode() called successfully (immediate).");
//             }
//             catch (System.Exception ex)
//             {
//                 Debug.LogError($"[CarCrashHandler] Exception while calling EndEpisode(): {ex}");
//             }

//             // allow future collisions to be handled in the next frame
//             StartCoroutine(ResetHasCrashedNextFrame());
//         }
//     }

//     private System.Collections.IEnumerator EndEpisodeNextFrame(CarAgent agent)
//     {
//         // wait until end of frame so ML-Agents/physics finish current callbacks
//         yield return new WaitForEndOfFrame();

//         if (agent == null)
//         {
//             Debug.LogWarning("[CarCrashHandler] Agent was destroyed before EndEpisodeNextFrame.");
//             hasCrashed = false; // ensure handler can respond later
//             yield break;
//         }

//         // clear physics just before signalling the agent
//         var rb = agent.GetComponent<Rigidbody>();
//         if (rb != null)
//         {
//             rb.linearVelocity = Vector3.zero;
//             rb.angularVelocity = Vector3.zero;
//         }

//         try
//         {
//             agent.EndEpisode();
//             Debug.Log("[CarCrashHandler] EndEpisode() called successfully (deferred).");
//         }
//         catch (System.Exception ex)
//         {
//             Debug.LogError($"[CarCrashHandler] Failed to EndEpisode safely: {ex}");
//         }

//         // allow future collisions to be handled
//         hasCrashed = false;
//     }


//     // Small coroutine used to reset the hasCrashed guard next frame
//     private System.Collections.IEnumerator ResetHasCrashedNextFrame()
//     {
//         yield return new WaitForEndOfFrame();
//         hasCrashed = false;
//     }

//     private bool IsForbidden(GameObject other)
//     {
//         for (int i = 0; i < forbiddenObjects.Count; ++i)
//         {
//             var f = forbiddenObjects[i];
//             if (f == null) continue;

//             if (other == f) return true;
//             if (other.transform.IsChildOf(f.transform)) return true;
//             if (f.transform.IsChildOf(other.transform)) return true;
//         }
//         return false;
//     }
// }

using UnityEngine;
using System.Collections.Generic;

[DisallowMultipleComponent]
public class CarCrashHandler : MonoBehaviour
{
    [Tooltip("Assign objects (with BoxCollider/Collider) that will cause the episode to end when touched.")]
    public List<GameObject> forbiddenObjects = new List<GameObject>();

    [Tooltip("If assigned, this agent will be signalled on crash. If null, will auto-find on THIS car.")]
    public CarAgent agentReference = null;

    [Tooltip("If true, EndEpisode will be delayed until the next frame to avoid sensor-reset race conditions.")]
    public bool delayEndEpisodeOneFrame = true;

    // Prevent repeated crash handling within the same physics step
    private bool hasCrashed = false;

    // CRITICAL FIX: Cache the agent reference on Awake
    private void Awake()
    {
        if (agentReference == null)
        {
            // Find agent on THIS car only (not globally in scene)
            agentReference = GetComponent<CarAgent>();
            
            if (agentReference == null)
            {
                agentReference = GetComponentInParent<CarAgent>();
            }
            
            if (agentReference == null)
            {
                agentReference = GetComponentInChildren<CarAgent>();
            }
            
            if (agentReference == null)
            {
                Debug.LogError($"[CarCrashHandler on {gameObject.name}] No CarAgent found on this GameObject or its parent/children! Crash handling will NOT work.");
            }
            else
            {
                Debug.Log($"[CarCrashHandler on {gameObject.name}] Auto-found CarAgent: {agentReference.gameObject.name}");
            }
        }
    }

    // Handle both physics collisions and trigger zones
    private void OnCollisionEnter(Collision collision) => HandleCollision(collision.gameObject);
    private void OnTriggerEnter(Collider other) => HandleCollision(other.gameObject);

    private void HandleCollision(GameObject other)
    {
        if (hasCrashed || other == null) return;
        if (!IsForbidden(other)) return;

        hasCrashed = true;
        Debug.Log($"[CarCrashHandler on {gameObject.name}] Car crashed into: {other.name}");

        // Use cached agent reference (found in Awake)
        if (agentReference == null)
        {
            Debug.LogWarning($"[CarCrashHandler on {gameObject.name}] No CarAgent reference! Cannot end episode.");
            hasCrashed = false;
            return;
        }

        // diagnostic: does agent look correctly configured?
        var bp = agentReference.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        Debug.Log($"[CarCrashHandler] Ending episode on agent '{agentReference.name}'. BehaviorParameters present: {(bp != null)}");

        if (delayEndEpisodeOneFrame)
        {
            // deferred path: do the safe EndEpisode on the next frame
            StartCoroutine(EndEpisodeNextFrame(agentReference));
        }
        else
        {
            // immediate path: clear physics, then end episode now
            var rb = agentReference.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }

            try
            {
                agentReference.EndEpisode();
                Debug.Log("[CarCrashHandler] EndEpisode() called successfully (immediate).");
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[CarCrashHandler] Exception while calling EndEpisode(): {ex}");
            }

            // allow future collisions to be handled in the next frame
            StartCoroutine(ResetHasCrashedNextFrame());
        }
    }

    private System.Collections.IEnumerator EndEpisodeNextFrame(CarAgent agent)
    {
        // wait until end of frame so ML-Agents/physics finish current callbacks
        yield return new WaitForEndOfFrame();

        if (agent == null)
        {
            Debug.LogWarning("[CarCrashHandler] Agent was destroyed before EndEpisodeNextFrame.");
            hasCrashed = false;
            yield break;
        }

        // clear physics just before signalling the agent
        var rb = agent.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        try
        {
            agent.EndEpisode();
            Debug.Log("[CarCrashHandler] EndEpisode() called successfully (deferred).");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[CarCrashHandler] Failed to EndEpisode safely: {ex}");
        }

        // allow future collisions to be handled
        hasCrashed = false;
    }

    // Small coroutine used to reset the hasCrashed guard next frame
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