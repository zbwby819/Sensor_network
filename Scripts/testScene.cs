using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class testScene : MonoBehaviour
{
    private int num;
    public GameObject change;
    public GameObject target;

    float obstacleCheckRadius = 1.5f;

    private bool c_flag1 = true;
    // Start is called before the first frame update
    void Start()
    {
        num = 0;
    }

    // Update is called once per frame
    void Update()
    {
        num++;
        if (num == 150)
        {
            if (change)
            {
                change.GetComponent<changeScene>().change_scene = true;
            }
            num = 0;

            float tx = UnityEngine.Random.Range(0f, 39f);
            float tz = UnityEngine.Random.Range(-39f, 0f);
            float rx = UnityEngine.Random.Range(0f, 39f);
            float rz = UnityEngine.Random.Range(-39f, 0f);
            while (c_flag1 == true)
            {
                c_flag1 = false;
                Vector3 position1 = new Vector3(tx, 0.75f, tz);
                Collider[] colliders1 = Physics.OverlapSphere(position1, obstacleCheckRadius);
                foreach (Collider col in colliders1)
                {
                    // If this collider is tagged "Obstacle"
                    if (col.tag == "Obstacle" || col.tag == "Sensor")
                    {
                        // Then this position is not a valid spawn position
                        c_flag1 = true;
                        tx = UnityEngine.Random.Range(0f, 39f);
                        tz = UnityEngine.Random.Range(-39f, 0f);
                    }
                }

            }
            target.transform.position = new Vector3(tx, 0.5f, tz);
        }
    }
}
