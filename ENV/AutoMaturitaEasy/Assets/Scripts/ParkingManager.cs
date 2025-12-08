// using UnityEngine;

// [DisallowMultipleComponent]
// public class ParkingManager : MonoBehaviour
// {
//     [Tooltip("Assign the single parking spot in the Inspector")]
//     public ParkingSpot singleSpot;
    
//     public ParkingSpot CurrentAssignedSpot { get; private set; }

//     public bool autoStartOnPlay = true;

//     [Tooltip("Enable to see manager debug logs. Disable in release to avoid allocations/log spam.")]
//     public bool debugLogs = true;

//     void Start()
//     {
//         // Auto-find the spot if not assigned
//         if (singleSpot == null)
//         {
//             singleSpot = FindFirstObjectByType<ParkingSpot>();
//             if (debugLogs && singleSpot != null) 
//                 Debug.Log($"[PM] Auto-found spot: {singleSpot.name}");
//         }

//         if (singleSpot == null)
//         {
//             Debug.LogError("[PM] No parking spot found! Please assign one in the Inspector.");
//             return;
//         }

//         if (autoStartOnPlay)
//             StartRound();
//     }

//     public void StartRound()
//     {
//         if (singleSpot == null)
//         {
//             if (debugLogs) Debug.LogWarning("[PM] No spot defined to start round.");
//             return;
//         }

//         if (debugLogs) Debug.Log("[PM] ===== STARTING NEW ROUND (Single Spot Mode) =====");

//         // Reset the spot
//         singleSpot.ResetSpot();

//         // Always assign this spot as the goal
//         singleSpot.isAssigned = true;
//         singleSpot.isGoal = false;
        
//         if (singleSpot.spotTrigger != null)
//             singleSpot.spotTrigger.enabled = true;

//         CurrentAssignedSpot = singleSpot;

//         if (debugLogs) 
//             Debug.Log($"[PM] Goal spot ready: {singleSpot.name}");
//     }

//     // Allow manual restart for testing
//     void Update()
//     {
//         if (Input.GetKeyDown(KeyCode.R))
//             StartRound();
//     }
// }

using UnityEngine;

[DisallowMultipleComponent]
public class ParkingManager : MonoBehaviour
{
    [Tooltip("Assign the single parking spot in the Inspector")]
    public ParkingSpot singleSpot;
    
    public ParkingSpot CurrentAssignedSpot { get; private set; }

    public bool autoStartOnPlay = true;

    [Tooltip("Enable to see manager debug logs. Disable in release to avoid allocations/log spam.")]
    public bool debugLogs = true;

    void Start()
    {
        // Auto-find the spot if not assigned
        if (singleSpot == null)
        {
            // CRITICAL FIX: Find spot only in THIS environment (as child of this manager's parent)
            // GetComponentInChildren searches THIS object and all descendants
            singleSpot = GetComponentInChildren<ParkingSpot>();
            
            // Fallback: if not found as child, try finding in parent's children
            if (singleSpot == null && transform.parent != null)
            {
                singleSpot = transform.parent.GetComponentInChildren<ParkingSpot>();
            }
            
            if (debugLogs && singleSpot != null) 
                Debug.Log($"[PM {gameObject.name}] Auto-found spot: {singleSpot.name} at {singleSpot.transform.position}");
        }

        if (singleSpot == null)
        {
            Debug.LogError($"[PM {gameObject.name}] No parking spot found! Please assign one in the Inspector or make sure ParkingSpot is in the same environment.");
            return;
        }

        // Verify the spot is in the same environment (safety check)
        if (transform.parent != null && !IsChildOfSameParent(singleSpot.transform))
        {
            Debug.LogError($"[PM {gameObject.name}] Found spot '{singleSpot.name}' but it's in a different environment! This will cause issues.");
        }

        if (autoStartOnPlay)
            StartRound();
    }

    public void StartRound()
    {
        if (singleSpot == null)
        {
            if (debugLogs) Debug.LogWarning($"[PM {gameObject.name}] No spot defined to start round.");
            return;
        }

        if (debugLogs) Debug.Log($"[PM {gameObject.name}] ===== STARTING NEW ROUND (Single Spot Mode) =====");

        // Reset the spot
        singleSpot.ResetSpot();

        // Always assign this spot as the goal
        singleSpot.isAssigned = true;
        singleSpot.isGoal = false;
        
        if (singleSpot.spotTrigger != null)
            singleSpot.spotTrigger.enabled = true;

        CurrentAssignedSpot = singleSpot;

        if (debugLogs) 
            Debug.Log($"[PM {gameObject.name}] Goal spot ready: {singleSpot.name} at {singleSpot.transform.position}");
    }

    // Helper method to verify spot is in same environment
    private bool IsChildOfSameParent(Transform spotTransform)
    {
        Transform myParent = transform.parent;
        Transform spotParent = spotTransform.parent;
        
        // Walk up the hierarchy to find if they share the same environment parent
        while (spotParent != null)
        {
            if (spotParent == myParent)
                return true;
            spotParent = spotParent.parent;
        }
        
        return false;
    }

    // Allow manual restart for testing
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
            StartRound();
    }
}