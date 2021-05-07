using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using UnityEngine;

public class testMove : MonoBehaviour
{
    public float speed = 10.0f;
    private Transform mmTransform;
    private Rigidbody mmRigidbody;

    // Use this for initialization
    void Start()
    {
        mmTransform = gameObject.GetComponent<Transform>();
        mmRigidbody = gameObject.GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        //使小球按照y轴正方向以每帧1的单位长度移动
        Vector3 position = new Vector3(-1, 0, 0);
        //mmRigidbody.AddRelativeForce(position * 1);
        //mmRigidbody.MovePosition(mmTransform.position + position);
        //transform.Translate(position, Space.World);
        mmRigidbody.velocity = position*10;
    }
    void OnCollisionEnter(Collision collision)
    {

        if (collision.gameObject.tag == "Obstacle")
        {
            print("1");
        }
    }

}


