using UnityEngine;

public class BirdEyeCamera : MonoBehaviour
{
    [Header("Camera Settings")]
    public Transform car; // Assign this in the Inspector
    public float height = 15f;
    public float distance = 10f;
    public float angle = 45f;
    public float smoothSpeed = 5f;
    
    [Header("Auto-Find Settings")]
    public bool autoFindCar = true;
    public string carTag = "Player";
    
    void Start()
    {
        // Try to find car automatically if not assigned
        if (car == null && autoFindCar)
        {
            FindCarAutomatically();
        }
    }
    
    void FindCarAutomatically()
    {
        GameObject carObject = GameObject.FindGameObjectWithTag(carTag);
        if (carObject != null)
        {
            car = carObject.transform;
            Debug.Log("Automatically found car: " + car.name);
        }
        else
        {
            Debug.LogWarning("Could not find car with tag: " + carTag);
            
            // Try to find any object with "car" in the name
            carObject = GameObject.Find("Car");
            if (carObject == null) carObject = GameObject.Find("car");
            
            if (carObject != null)
            {
                car = carObject.transform;
                Debug.Log("Found car by name: " + car.name);
            }
            else
            {
                Debug.LogError("Could not find any car object automatically!");
            }
        }
    }
    
    void LateUpdate()
    {
        if (car == null)
        {
            // Try to find car again if it's still null
            if (autoFindCar) FindCarAutomatically();
            
            if (car == null)
            {
                Debug.LogWarning("Car reference is still null. Camera cannot follow.");
                return;
            }
        }
        
        // Calculate desired camera position
        Vector3 carPosition = car.position;
        
        // Convert angle to radians
        float angleRad = angle * Mathf.Deg2Rad;
        
        // Calculate offset based on angle and distance
        float xOffset = -distance * Mathf.Cos(angleRad);
        float zOffset = -distance * Mathf.Sin(angleRad);
        
        Vector3 desiredPosition = new Vector3(
            carPosition.x + xOffset,
            carPosition.y + height,
            carPosition.z + zOffset
        );
        
        // Smoothly move the camera to the desired position
        transform.position = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);
        
        // Always look at the car
        transform.LookAt(carPosition);
    }
    
    // Method to manually set the car reference
    public void SetCar(Transform newCar)
    {
        car = newCar;
        Debug.Log("Camera now following: " + car.name);
    }
}