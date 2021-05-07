using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using System.Collections.Specialized;
using System.Runtime.InteropServices;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class dynaSensor : Agent
{
    private Rigidbody rBody;
    private Transform mmTransform;

    public GameObject target;
    private GameObject target_s1;
    private GameObject target_s2;
    private GameObject target_s3;
    private GameObject target_s4;
    private GameObject target_s5;
    private GameObject target_s6;
    private GameObject target_s7;
    private GameObject target_s8;
    private GameObject target_s9;
    private GameObject target_s10;
    private GameObject target_s11;
    private GameObject target_s12;
    private GameObject target_s13;
    private GameObject target_s14;
    private GameObject target_s15;
    private GameObject target_s16;
    private GameObject target_s17;
    private GameObject target_s18;
    private GameObject target_s19;
    private GameObject target_s20;
    private GameObject target_s21;
    private GameObject target_s22;

    public GameObject root_env;
    public GameObject all_sensors;
    private GameObject prev_env;
    private GameObject current_env;
    private string[] all_env = new string[18] { "Env_1", "Env_2", "Env_3", "Env_4", "Env_5", "Env_6", "Env_7", "Env_8",
                                                "Env_9", "Env_10", "Env_11", "Env_12", "Env_13", "Env_14", "Env_15", "Env_16", "Env_17", "Env_18" };

    public int min_sensor = 9;
    public int max_sensor = 17;
    public int num_sensor = 9;
    public int select_env = 0;

    private int old_env = 0;
    public bool change_scene = false;

    int num = 0;
    int steps = 0;
    public float dull_action = 0f; 
    static int loc_length = 100;
    float[] t_locx = new float[loc_length], t_locy = new float[loc_length], r_locx = new float[loc_length], r_locy = new float[loc_length];

    public float random_x = 39f;
    public float random_z = 39f;
    float obstacleCheckRadius = 2.0f;
    // Start is called before the first frame update
    void Start()
    {
        mmTransform = gameObject.GetComponent<Transform>();
        rBody = gameObject.GetComponent<Rigidbody>();
        print(string.Format("current env is {0}", select_env + 1));
        current_env = root_env.transform.Find(all_env[select_env]).gameObject;
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
        
        num_sensor = UnityEngine.Random.Range(min_sensor, max_sensor);
        print(string.Format("New episode begins! number of sensors are {0}", num_sensor));
        //num_sensor = 9;

        if (change_scene)
        {
            select_env = UnityEngine.Random.Range(0, all_env.Length);
            while (select_env == old_env)
            {
                select_env = UnityEngine.Random.Range(0, all_env.Length);
            }
            //print("prepare load env");
            current_env = root_env.transform.Find(all_env[select_env]).gameObject;
            current_env.SetActive(true);
            prev_env = root_env.transform.Find(all_env[old_env]).gameObject;
            prev_env.SetActive(false);
            old_env = select_env;
            print(string.Format("change scene to {0}", select_env + 1));
        }
        steps = 1;

        target.transform.position = GenerateSensor(random_x, random_z);
        //target.transform.position = new Vector3(1f, 0.5f, 1f);
        this.transform.position = GenerateSensor(random_x, random_z);
        //this.transform.position = new Vector3(3f, 0.5f, 3f);
        
        //this.GetComponent<saveSensor_2>().num = steps;
        //this.GetComponent<saveSensor_2>().camsave(steps);

        target_s1 = current_env.transform.Find("Sensor_1").gameObject;
        //target_s1.GetComponent<saveSensor_2>().num = steps;
        //target_s1.GetComponent<saveSensor_2>().camsave(steps);

        target_s2 = current_env.transform.Find("Sensor_2").gameObject;
        //target_s2.GetComponent<saveSensor_2>().num = steps;
        //target_s2.GetComponent<saveSensor_2>().camsave(steps);

        target_s3 = current_env.transform.Find("Sensor_3").gameObject;
        //target_s3.GetComponent<saveSensor_2>().num = steps;
        //target_s3.GetComponent<saveSensor_2>().camsave(steps);

        target_s4 = current_env.transform.Find("Sensor_4").gameObject;
        //target_s4.GetComponent<saveSensor_2>().num = steps;
        //target_s4.GetComponent<saveSensor_2>().camsave(steps);

        target_s5 = current_env.transform.Find("Sensor_5").gameObject;
        //target_s5.GetComponent<saveSensor_2>().num = steps;
        //target_s5.GetComponent<saveSensor_2>().camsave(steps);

        target_s6 = current_env.transform.Find("Sensor_6").gameObject;
        //target_s6.GetComponent<saveSensor_2>().num = steps;
        //target_s6.GetComponent<saveSensor_2>().camsave(steps);

        target_s7 = current_env.transform.Find("Sensor_7").gameObject;
        //target_s7.GetComponent<saveSensor_2>().num = steps;
        //target_s7.GetComponent<saveSensor_2>().camsave(steps);

        target_s8 = current_env.transform.Find("Sensor_8").gameObject;
        //target_s8.GetComponent<saveSensor_2>().num = steps;
        //target_s8.GetComponent<saveSensor_2>().camsave(steps);

        target_s9 = current_env.transform.Find("Sensor_9").gameObject;
        //target_s9.GetComponent<saveSensor_2>().num = steps;
        //target_s9.GetComponent<saveSensor_2>().camsave(steps);

        if (num_sensor >= 10)
        {
            target_s10 = all_sensors.transform.Find("Sensor_10").gameObject;
            target_s10.SetActive(true);
            target_s10.transform.position = GenerateSensor(random_x, random_z);
            //target_s10.GetComponent<saveSensor_2>().num = steps;
            //target_s10.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s10 = all_sensors.transform.Find("Sensor_10").gameObject;
            target_s10.SetActive(false);
        }

        if (num_sensor >= 11)
        {
            target_s11 = all_sensors.transform.Find("Sensor_11").gameObject;
            target_s11.SetActive(true);
            target_s11.transform.position = GenerateSensor(random_x, random_z);
            //target_s11.GetComponent<saveSensor_2>().num = steps;
            //target_s11.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s11 = all_sensors.transform.Find("Sensor_11").gameObject;
            target_s11.SetActive(false);
        }

        if (num_sensor >= 12)
        {
            target_s12 = all_sensors.transform.Find("Sensor_12").gameObject;
            target_s12.SetActive(true);
            target_s12.transform.position = GenerateSensor(random_x, random_z);
            //target_s12.GetComponent<saveSensor_2>().num = steps;
            //target_s12.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s12 = all_sensors.transform.Find("Sensor_12").gameObject;
            target_s11.SetActive(false);
        }

        if (num_sensor >= 13)
        {
            target_s13 = all_sensors.transform.Find("Sensor_13").gameObject;
            target_s13.SetActive(true);
            target_s13.transform.position = GenerateSensor(random_x, random_z);
            //target_s13.GetComponent<saveSensor_2>().num = steps;
            //target_s13.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s13 = all_sensors.transform.Find("Sensor_13").gameObject;
            target_s13.SetActive(false);
        }

        if (num_sensor >= 14)
        {
            target_s14 = all_sensors.transform.Find("Sensor_14").gameObject;
            target_s14.SetActive(true);
            target_s14.transform.position = GenerateSensor(random_x, random_z);
            //target_s14.GetComponent<saveSensor_2>().num = steps;
            //target_s14.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s14 = all_sensors.transform.Find("Sensor_14").gameObject;
            target_s14.SetActive(false);
        }

        if (num_sensor >= 15)
        {
            target_s15 = all_sensors.transform.Find("Sensor_15").gameObject;
            target_s15.SetActive(true);
            target_s15.transform.position = GenerateSensor(random_x, random_z);
            //target_s15.GetComponent<saveSensor_2>().num = steps;
            //target_s15.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s15 = all_sensors.transform.Find("Sensor_15").gameObject;
            target_s15.SetActive(false);
        }

        if (num_sensor >= 16)
        {
            target_s16 = all_sensors.transform.Find("Sensor_16").gameObject;
            target_s16.SetActive(true);
            target_s16.transform.position = GenerateSensor(random_x, random_z);
            //target_s16.GetComponent<saveSensor_2>().num = steps;
            //target_s16.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s16 = all_sensors.transform.Find("Sensor_16").gameObject;
            target_s16.SetActive(false);
        }

        if (num_sensor >= 17)
        {
            target_s17 = all_sensors.transform.Find("Sensor_17").gameObject;
            target_s17.SetActive(true);
            target_s17.transform.position = GenerateSensor(random_x, random_z);
            //target_s17.GetComponent<saveSensor_2>().num = steps;
            //target_s17.GetComponent<saveSensor_2>().camsave(steps);
        }
        else
        {
            target_s17 = all_sensors.transform.Find("Sensor_17").gameObject;
            target_s17.SetActive(false);
        }

        //steps++;
        num++;
    }

    public Vector3 GenerateSensor(float x_size = 39f, float z_size = 39f, float obstacleCheckRadius = 3.0f)
    {

        float rx = UnityEngine.Random.Range(0f, x_size);
        float rz = UnityEngine.Random.Range(-z_size, 0f);
        bool c_flag1 = true;

        while (c_flag1 == true)
        {
            c_flag1 = false;
            Vector3 position1 = new Vector3(rx, 0.5f, rz);
            Collider[] colliders1 = Physics.OverlapSphere(position1, obstacleCheckRadius);
            foreach (Collider col in colliders1)
            {
                // If this collider is tagged "Obstacle"
                if (col.tag == "Obstacle" || col.tag == "Sensor")
                {
                    // Then this position is not a valid spawn position
                    c_flag1 = true;
                    rx = UnityEngine.Random.Range(0f, x_size);
                    rz = UnityEngine.Random.Range(-z_size, 0f);
                }
            }
        }
        return new Vector3(rx, 0.5f, rz);
    }

    public void SaveLoc(Vector3 _target_pos, string save_path, int steps)
    {
        //FileStream fs = new FileStream(Application.dataPath + "/target_loc.txt", FileMode.Append);
        FileStream fs = new FileStream(save_path, FileMode.Append);
        byte[] bytes = new UTF8Encoding().GetBytes(_target_pos.ToString() + string.Format("{0}\r\n", steps));
        fs.Write(bytes, 0, bytes.Length);
        fs.Close();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions & Agent velocity
        sensor.AddObservation(target.transform.position);
        sensor.AddObservation(this.transform.position);
        sensor.AddObservation(select_env);
        sensor.AddObservation(num_sensor);
        sensor.AddObservation(steps);

        sensor.AddObservation(target_s1.transform.localPosition);
        sensor.AddObservation(target_s2.transform.localPosition);
        sensor.AddObservation(target_s3.transform.localPosition);
        sensor.AddObservation(target_s4.transform.localPosition);
        sensor.AddObservation(target_s5.transform.localPosition);
        sensor.AddObservation(target_s6.transform.localPosition);
        sensor.AddObservation(target_s7.transform.localPosition);
        sensor.AddObservation(target_s8.transform.localPosition);
        sensor.AddObservation(target_s9.transform.localPosition);
        if (target_s10.activeSelf == true)
        {
            sensor.AddObservation(target_s10.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s11.activeSelf == true)
        {
            sensor.AddObservation(target_s11.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s12.activeSelf == true)
        {
            sensor.AddObservation(target_s12.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s13.activeSelf == true)
        {
            sensor.AddObservation(target_s13.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s14.activeSelf == true)
        {
            sensor.AddObservation(target_s14.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s15.activeSelf == true)
        {
            sensor.AddObservation(target_s15.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s16.activeSelf == true)
        {
            sensor.AddObservation(target_s16.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

        if (target_s17.activeSelf == true)
        {
            sensor.AddObservation(target_s17.transform.localPosition);
        }
        else
        {
            sensor.AddObservation(new Vector3(0f, 0f, 0f));
        }

    }

    public override void OnActionReceived(float[] vectorAction)
    {
        //print("receieve acion!");
        dull_action = vectorAction[0];
        //dull_action = 1f;
        if (dull_action == 1f)
        {
            while (steps <= 64)
            {
                target.transform.position = GenerateSensor(random_x, random_z);
                //target.transform.position = new Vector3(1f, 0.5f, 1f);
                this.transform.position = GenerateSensor(random_x, random_z);
                //this.transform.position = new Vector3(3f, 0.5f, 3f);

                SaveLoc(target.transform.position, string.Format("C:/Sensor_network/train_data1101/epoch_{0}_target.txt", num), steps);
                SaveLoc(this.transform.position, string.Format("C:/Sensor_network/train_data1101/epoch_{0}_robot.txt", num), steps);

                this.GetComponent<saveSensor_2>().num = steps;
                this.GetComponent<saveSensor_2>().camsave(steps);

                target_s1.GetComponent<saveOnce>().is_save = true;
                target_s1.GetComponent<saveOnce>().save_img(steps);

                target_s2.GetComponent<saveOnce>().is_save = true;
                target_s2.GetComponent<saveOnce>().save_img(steps);

                target_s3.GetComponent<saveOnce>().is_save = true;
                target_s3.GetComponent<saveOnce>().save_img(steps);

                target_s4.GetComponent<saveOnce>().is_save = true;
                target_s4.GetComponent<saveOnce>().save_img(steps);

                target_s5.GetComponent<saveOnce>().is_save = true;
                target_s5.GetComponent<saveOnce>().save_img(steps);

                target_s6.GetComponent<saveOnce>().is_save = true;
                target_s6.GetComponent<saveOnce>().save_img(steps);

                target_s7.GetComponent<saveOnce>().is_save = true;
                target_s7.GetComponent<saveOnce>().save_img(steps);

                target_s8.GetComponent<saveOnce>().is_save = true;
                target_s8.GetComponent<saveOnce>().save_img(steps);

                target_s9.GetComponent<saveOnce>().is_save = true;
                target_s9.GetComponent<saveOnce>().save_img(steps);

                if (num_sensor >= 10)
                {
                    target_s10.GetComponent<saveSensor_2>().num = steps;
                    target_s10.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 11)
                {
                    target_s11.GetComponent<saveSensor_2>().num = steps;
                    target_s11.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 12)
                {
                    target_s12.GetComponent<saveSensor_2>().num = steps;
                    target_s12.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 13)
                {
                    target_s13.GetComponent<saveSensor_2>().num = steps;
                    target_s13.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 14)
                {
                    target_s14.GetComponent<saveSensor_2>().num = steps;
                    target_s14.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 15)
                {
                    target_s15.GetComponent<saveSensor_2>().num = steps;
                    target_s15.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 16)
                {
                    target_s16.GetComponent<saveSensor_2>().num = steps;
                    target_s16.GetComponent<saveSensor_2>().camsave(steps);
                }

                if (num_sensor >= 17)
                {
                    target_s17.GetComponent<saveSensor_2>().num = steps;
                    target_s17.GetComponent<saveSensor_2>().camsave(steps);
                }
                steps++;

            }
        }
        if (dull_action == 2f )
        {
            print("End episode");
            EndEpisode();
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        if (Input.GetKey(KeyCode.W))
        {
            rBody.AddForce(Vector3.forward * 100, ForceMode.Force);
        }
        if (Input.GetKey(KeyCode.S))
        {
            rBody.AddForce(Vector3.back * 100, ForceMode.Force);
        }
        if (Input.GetKey(KeyCode.A))
        {
            rBody.AddForce(Vector3.left * 100, ForceMode.Force);
        }
        if (Input.GetKey(KeyCode.D))
        {
            rBody.AddForce(Vector3.right * 100, ForceMode.Force);
        }
    }
}
