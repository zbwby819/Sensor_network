using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Specialized;
using System.Runtime.InteropServices;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class testRobot : Agent
{
    private Rigidbody rBody;
    private Transform mmTransform;

    public GameObject target;
    public bool save_robot_loc = false;
    public bool is_test = false;
    //bool is_collide = false;
    public int num = 1;

    public float speed = 50.0f;

    // Start is called before the first frame update
    void Start()
    {
        if (is_test == true)
        {
            Time.timeScale = 0.1f;
        }
        mmTransform = gameObject.GetComponent<Transform>();
        rBody = gameObject.GetComponent<Rigidbody>();

    }

    public void save_pos(Vector3 _robot_pos, int num)
    {
        FileStream fs2 = new FileStream($"C:/Cambridge/Sensor_network/training/robot_loc.txt", FileMode.Append);
        byte[] bytes2 = new UTF8Encoding().GetBytes(_robot_pos.ToString() + string.Format("{0}\r\n", num));
        fs2.Write(bytes2, 0, bytes2.Length);
        fs2.Close();
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;

        num = 1;
        if (save_robot_loc)
        {
            save_pos(this.transform.position, num);
        }
        num++;
 
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent positions & Agent velocity
        sensor.AddObservation(this.transform.position);
        sensor.AddObservation(rBody.velocity);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (save_robot_loc)
        {
            save_pos(this.transform.position, num);
        }

        Vector3 cur_loc = this.transform.localPosition;

        float forward_x = actionBuffers.ContinuousActions[1] * speed;
        float forward_z = -actionBuffers.ContinuousActions[0] * speed;

        Vector3 newposition1 = new Vector3(forward_x, 0, forward_z);
        //rBody.AddRelativeForce(newposition1 * speed);
        //rBody.MovePosition(mmTransform.localPosition + newposition1);
        //this.transform.Translate(newposition1, Space.Self);
        rBody.velocity = (newposition1);
        
        float distanceToTarget = Vector3.Distance(this.transform.position, target.transform.position);
        // Reached target

        if (distanceToTarget < 1.0f)
        {
            SetReward(10.0f);
            EndEpisode();
        }
        SetReward(-0.01f);

        num++;
    }
}
