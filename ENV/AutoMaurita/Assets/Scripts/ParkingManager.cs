using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ParkingManager : MonoBehaviour
{
    public ParkingSpot[] spots;
    public int minFreeSpots = 1;
    public int maxFreeSpots = 3;
    public ParkingSpot CurrentAssignedSpot { get; private set; }

    public bool useDebugForceHide = true;
    public bool autoStartOnPlay = true;
    public bool debugLogs = true;

    void Start()
    {
        if (spots == null || spots.Length == 0)
        {
            Transform environmentRoot = transform.parent != null ? transform.parent : transform;
            
            ParkingSpot[] found = environmentRoot.GetComponentsInChildren<ParkingSpot>();
            System.Array.Sort(found, (a, b) => string.CompareOrdinal(a.name, b.name));
            spots = found;

            if (debugLogs) 
                Debug.Log($"[PM {gameObject.name}] Auto-found {spots.Length} spots in environment: {environmentRoot.name}");
        }

        if (transform.parent != null && spots != null)
        {
            foreach (var spot in spots)
            {
                if (spot != null && !IsInSameEnvironment(spot.transform))
                {
                    Debug.LogError($"[PM {gameObject.name}] Spot '{spot.name}' is NOT in the same environment!");
                }
            }
        }

        if (autoStartOnPlay)
            StartRound();
    }

    private bool IsInSameEnvironment(Transform other)
    {
        Transform myRoot = transform.parent != null ? transform.parent : transform;
        Transform otherParent = other.parent;
        
        while (otherParent != null)
        {
            if (otherParent == myRoot)
                return true;
            otherParent = otherParent.parent;
        }
        
        return false;
    }

    public void StartRound()
    {
        if (spots == null || spots.Length == 0)
        {
            if (debugLogs) Debug.LogWarning($"[PM {gameObject.name}] No spots to start round.");
            return;
        }

        if (debugLogs) Debug.Log($"[PM {gameObject.name}] ===== STARTING NEW ROUND =====");

        for (int i = 0; i < spots.Length; ++i)
            spots[i].ResetSpot();

        int freeCount = Random.Range(minFreeSpots, maxFreeSpots + 1);
        freeCount = Mathf.Clamp(freeCount, 1, Mathf.Max(1, spots.Length - 1));
        if (debugLogs) Debug.Log($"[PM {gameObject.name}] Freeing {freeCount} spots");

        List<int> indices = new List<int>(spots.Length);
        for (int i = 0; i < spots.Length; ++i) indices.Add(i);
        Shuffle(indices);

        List<int> freed = new List<int>(freeCount);
        for (int i = 0; i < freeCount && i < indices.Count; ++i) freed.Add(indices[i]);

        for (int i = 0; i < freed.Count; ++i)
        {
            int idx = freed[i];
            var s = spots[idx];
            if (useDebugForceHide)
                s.FreeSpot_DebugForceHide();
            else
                s.FreeSpot();
        }

        if (freed.Count > 0)
        {
            int pick = Random.Range(0, freed.Count);
            int goalIdx = freed[pick];

            if (debugLogs) Debug.Log($"[PM {gameObject.name}] Goal: idx={goalIdx}, spot={spots[goalIdx].name}");

            for (int i = 0; i < spots.Length; ++i)
            {
                bool isAssigned = (i == goalIdx);
                spots[i].isAssigned = isAssigned;
                spots[i].isGoal = false;
                if (spots[i].spotTrigger != null)
                    spots[i].spotTrigger.enabled = isAssigned;
            }

            CurrentAssignedSpot = spots[goalIdx];
        }
        else
        {
            Debug.LogError($"[PM {gameObject.name}] No freed spots!");
            CurrentAssignedSpot = null;
        }
    }

    void Shuffle<T>(List<T> list)
    {
        for (int i = 0; i < list.Count; ++i)
        {
            int j = Random.Range(i, list.Count);
            T tmp = list[i];
            list[i] = list[j];
            list[j] = tmp;
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
            StartRound();
    }
}