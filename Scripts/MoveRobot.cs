using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using System.Collections.Specialized;
using System.Runtime.InteropServices;
using System.IO;
using System;
using System.Text;
using UnityEngine;

public class MoveRobot : Agent
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
    private GameObject prev_env;
    private GameObject current_env;
    private string[] all_env = new string[24] { "Env_1", "Env_2", "Env_3", "Env_4", "Env_5", "Env_6", "Env_7", "Env_8", 
                                                "Env_9", "Env_10", "Env_11", "Env_12", "Env_13", "Env_14", "Env_15", "Env_16", "Env_17",
                                                "Env_18", "Env_19", "Env_20", "Env_21", "Env_22", "Env_23", "Env_24" };

    public int num_sensor = 22;
    public int select_env = 0;
    private int old_env = 0;
    public bool change_scene = false;
    public bool read_loc = false;
    public bool save_loc = false;
    public bool save_each_step = false;
    public bool save_robot_obs = false;
    public bool is_test = false;
    public bool is_same_random = false;
    //bool is_collide = false;
    int num = 0;
    int steps = 1;
    static int loc_length = 100;
    float[] t_locx = new float[loc_length], t_locy = new float[loc_length], r_locx = new float[loc_length], r_locy = new float[loc_length];

    public float speed = 50.0f;
    public float random_x = 59f;
    public float random_z = 59f;
    // Start is called before the first frame update
    void Start()
    {
        if (is_test == true)
        {
            Time.timeScale = 0.1f;
        } 
        mmTransform = gameObject.GetComponent<Transform>();
        rBody = gameObject.GetComponent<Rigidbody>();
        current_env = root_env.transform.Find(all_env[select_env]).gameObject;
        print(string.Format("current env is {0}", select_env+1));
        if (read_loc == true)
        {
            (t_locx, t_locy, r_locx, r_locy) = Loadloc();
        }
        if (is_same_random==false)
        {
            UnityEngine.Random.InitState((int)System.DateTime.Now.Ticks);
        }
    }

    public Tuple<float[], float[], float[], float[]> Loadloc()
    {
        int f_length = 8000;
        //FileMode.Open打开路径下的save.text文件
        string t_path = $"C:/Sensor_network/res1101/env_{select_env + 1}_target.txt";
        FileStream fs = new FileStream(t_path, FileMode.Open);

        byte[] bytes = new byte[f_length];
        fs.Read(bytes, 0, f_length);
        string s = new UTF8Encoding().GetString(bytes);

        string r_path = $"C:/Sensor_network/res1101/env_{select_env + 1}_robot.txt";
        FileStream fs2 = new FileStream(r_path,  FileMode.Open);
        byte[] bytes2 = new byte[f_length];
        //fs2.Read(bytes2, 0, fs2.Length);
        fs2.Read(bytes2, 0, f_length);
        string s2 = new UTF8Encoding().GetString(bytes2);

        string[] t_loc_string = s.Split('\n');
        string[] r_loc_string = s2.Split('\n');

        float[] t_loc_x = new float[loc_length];
        float[] t_loc_y = new float[loc_length];
        float[] r_loc_x = new float[loc_length];
        float[] r_loc_y = new float[loc_length];

        for (int i = 0; i < loc_length; i++)
        {
            //Debug.Log(t_loc_string[i]);
            string[] sArray = t_loc_string[i].Split(new char[3] { '(', ',', ')' });
            //Debug.Log((sArray[1], sArray[3]));
            t_loc_x[i] = float.Parse(sArray[1]);
            t_loc_y[i] = float.Parse(sArray[3]);
            //Debug.Log(t_loc_x[i]);
            //Debug.Log(t_loc_y[i]);
            string[] sArray2 = r_loc_string[i].Split(new char[3] { '(', ',', ')' });
            r_loc_x[i] = float.Parse(sArray2[1]);
            r_loc_y[i] = float.Parse(sArray2[3]);
        }
        Tuple<float[], float[], float[], float[]> all_loc = new Tuple<float[], float[], float[], float[]>(t_loc_x, t_loc_y, r_loc_x, r_loc_y);
        return all_loc;
    }

    public void save_pos(Vector3 _target_pos, Vector3 _robot_pos, int num)
    {
        //FileStream fs = new FileStream(Application.dataPath + "/target_loc.txt", FileMode.Append);
        FileStream fs = new FileStream($"C:/Sensor_network/res1101/env_{select_env + 1}_target.txt", FileMode.Append);
        byte[] bytes = new UTF8Encoding().GetBytes(_target_pos.ToString() + string.Format("{0}\r\n", num));
        fs.Write(bytes, 0, bytes.Length);
        fs.Close();

        FileStream fs2 = new FileStream($"C:/Sensor_network/res1101/env_{select_env + 1}_robot.txt", FileMode.Append);
        byte[] bytes2 = new UTF8Encoding().GetBytes(_robot_pos.ToString() + string.Format("{0}\r\n", num));
        fs2.Write(bytes2, 0, bytes2.Length);
        fs2.Close();
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;

        if (change_scene)
        {
            select_env = UnityEngine.Random.Range(0, all_env.Length);
            while (select_env == old_env)
            {
                select_env = UnityEngine.Random.Range(0, all_env.Length);
            }
            print("prepare load env");
            current_env = root_env.transform.Find(all_env[select_env]).gameObject;
            current_env.SetActive(true);
            prev_env = root_env.transform.Find(all_env[old_env]).gameObject;
            prev_env.SetActive(false);
            old_env = select_env;
            print(string.Format("change scene to {0}", select_env + 1));
        }
        steps = 1;

        this.GetComponent<saveOnce>().is_save = true;
        this.GetComponent<saveOnce>().save_img(steps);

        target_s1 = current_env.transform.Find("Sensor_1").gameObject;
        //target_s1.GetComponent<saveOnce>().is_save = true;
        //target_s1.GetComponent<saveOnce>().save_img(steps);

        target_s2 = current_env.transform.Find("Sensor_2").gameObject;
        //target_s2.GetComponent<saveOnce>().is_save = true;
        //target_s2.GetComponent<saveOnce>().save_img(steps);

        target_s3 = current_env.transform.Find("Sensor_3").gameObject;
        //target_s3.GetComponent<saveOnce>().is_save = true;
        //target_s3.GetComponent<saveOnce>().save_img(steps);

        target_s4 = current_env.transform.Find("Sensor_4").gameObject;
        //target_s4.GetComponent<saveOnce>().is_save = true;
        //target_s4.GetComponent<saveOnce>().save_img(steps);

        target_s5 = current_env.transform.Find("Sensor_5").gameObject;
        //target_s5.GetComponent<saveOnce>().is_save = true;
        //target_s5.GetComponent<saveOnce>().save_img(steps);

        target_s6 = current_env.transform.Find("Sensor_6").gameObject;
        //target_s6.GetComponent<saveOnce>().is_save = true;
        //target_s6.GetComponent<saveOnce>().save_img(steps);

        target_s7 = current_env.transform.Find("Sensor_7").gameObject;
        //target_s7.GetComponent<saveOnce>().is_save = true;
        //target_s7.GetComponent<saveOnce>().save_img(steps);

        target_s8 = current_env.transform.Find("Sensor_8").gameObject;
        //target_s8.GetComponent<saveOnce>().is_save = true;
        //target_s8.GetComponent<saveOnce>().save_img(steps);

        target_s9 = current_env.transform.Find("Sensor_9").gameObject;
        //target_s9.GetComponent<saveOnce>().is_save = true;
        //target_s9.GetComponent<saveOnce>().save_img(steps);

        if (num_sensor >=10)
        {
            target_s10 = current_env.transform.Find("Sensor_10").gameObject;
            target_s10.GetComponent<saveOnce>().is_save = true;
            target_s10.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 11)
        {
            target_s11 = current_env.transform.Find("Sensor_11").gameObject;
            target_s11.GetComponent<saveOnce>().is_save = true;
            target_s11.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 12)
        {
            target_s12 = current_env.transform.Find("Sensor_12").gameObject;
            target_s12.GetComponent<saveOnce>().is_save = true;
            target_s12.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 13)
        {
            target_s13 = current_env.transform.Find("Sensor_13").gameObject;
            target_s13.GetComponent<saveOnce>().is_save = true;
            target_s13.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 14)
        {
            target_s14 = current_env.transform.Find("Sensor_14").gameObject;
            target_s14.GetComponent<saveOnce>().is_save = true;
            target_s14.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 15)
        {
            target_s15 = current_env.transform.Find("Sensor_15").gameObject;
            target_s15.GetComponent<saveOnce>().is_save = true;
            target_s15.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 16)
        {
            target_s16 = current_env.transform.Find("Sensor_16").gameObject;
            target_s16.GetComponent<saveOnce>().is_save = true;
            target_s16.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 17)
        {
            target_s17 = current_env.transform.Find("Sensor_17").gameObject;
            target_s17.GetComponent<saveOnce>().is_save = true;
            target_s17.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 18)
        {
            target_s18 = current_env.transform.Find("Sensor_18").gameObject;
            target_s18.GetComponent<saveOnce>().is_save = true;
            target_s18.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 19)
        {
            target_s19 = current_env.transform.Find("Sensor_19").gameObject;
            target_s19.GetComponent<saveOnce>().is_save = true;
            target_s19.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 20)
        {
            target_s20 = current_env.transform.Find("Sensor_20").gameObject;
            target_s20.GetComponent<saveOnce>().is_save = true;
            target_s20.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 21)
        {
            target_s21 = current_env.transform.Find("Sensor_21").gameObject;
            target_s21.GetComponent<saveOnce>().is_save = true;
            target_s21.GetComponent<saveOnce>().save_img(steps);
        }

        if (num_sensor >= 22)
        {
            target_s22 = current_env.transform.Find("Sensor_22").gameObject;
            target_s22.GetComponent<saveOnce>().is_save = true;
            target_s22.GetComponent<saveOnce>().save_img(steps);
        }

        steps++;

        float obstacleCheckRadius = 2.0f;
        float tx = UnityEngine.Random.Range(0f, random_x);
        float tz = UnityEngine.Random.Range(-random_z, 0f);
        float rx = UnityEngine.Random.Range(0f, random_x);
        float rz = UnityEngine.Random.Range(-random_z, 0f);
        bool c_flag1 = true;
        bool c_flag2 = true;

        if (read_loc == false)
        {
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
                        tx = UnityEngine.Random.Range(0f, random_x);
                        tz = UnityEngine.Random.Range(-random_z, 0f);
                    }
                }

            }
            target.transform.position = new Vector3(tx, 0.5f, tz);
            //target.transform.position = new Vector3(13.5f, 0.75f, -16f);

            while (c_flag2 == true)
            {
                c_flag2 = false;
                Vector3 position2 = new Vector3(rx, 0.5f, rz);
                Collider[] colliders2 = Physics.OverlapSphere(position2, obstacleCheckRadius);
                foreach (Collider col in colliders2)
                {
                    // If this collider is tagged "Obstacle"
                    float dis2target = (float)Math.Pow((rx - tx), 2) + (float)Math.Pow((rz - tz), 2);
                    if (col.tag == "Obstacle" || col.tag == "Sensor" || dis2target <= 25.0f)
                    {
                        // Then this position is not a valid spawn position
                        c_flag2 = true;
                        rx = UnityEngine.Random.Range(0f, random_x);
                        rz = UnityEngine.Random.Range(-random_z, 0f);
                    }
                }
            }
            this.transform.position = new Vector3(rx, 0.5f, rz);
            //this.transform.position = new Vector3(54.7f, 0.5f, -30.13f);
            //print("environment is reset!");
        }
        if (read_loc == true)
        {
            this.transform.position = new Vector3(r_locy[num], 0.5f, -r_locx[num]);
            target.transform.position = new Vector3(t_locy[num], 0.5f, -t_locx[num]);
        }

        if (save_loc == true)
        {   
            save_pos(new Vector3 (-tz, 0.5f, tx), new Vector3 (-rz, 0.5f, rx),  num);
        }
        num++;
        //is_collide = false;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions & Agent velocity
        sensor.AddObservation(target.transform.position);
        sensor.AddObservation(this.transform.position);
        sensor.AddObservation(rBody.velocity);
        //sensor.AddObservation(this.transform.position);
        sensor.AddObservation(select_env);
        //sensor.AddObservation(is_collide);
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        if (save_robot_obs == true)
        {
            this.GetComponent<saveOnce>().is_save = true;
            this.GetComponent<saveOnce>().save_img(steps);
        }

        if (save_each_step == true)
        {
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
                target_s10.GetComponent<saveOnce>().is_save = true;
                target_s10.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 11)
            {
                target_s11.GetComponent<saveOnce>().is_save = true;
                target_s11.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 12)
            {
                target_s12.GetComponent<saveOnce>().is_save = true;
                target_s12.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 13)
            {
                target_s13.GetComponent<saveOnce>().is_save = true;
                target_s13.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 14)
            {
                target_s14.GetComponent<saveOnce>().is_save = true;
                target_s14.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 15)
            {
                target_s15.GetComponent<saveOnce>().is_save = true;
                target_s15.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 16)
            {
                target_s16.GetComponent<saveOnce>().is_save = true;
                target_s16.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 17)
            {
                target_s17.GetComponent<saveOnce>().is_save = true;
                target_s17.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 18)
            {
                target_s18.GetComponent<saveOnce>().is_save = true;
                target_s18.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 19)
            {
                target_s19.GetComponent<saveOnce>().is_save = true;
                target_s19.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 20)
            {
                target_s20.GetComponent<saveOnce>().is_save = true;
                target_s20.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 21)
            {
                target_s21.GetComponent<saveOnce>().is_save = true;
                target_s21.GetComponent<saveOnce>().save_img(steps);
            }

            if (num_sensor >= 22)
            {
                target_s22.GetComponent<saveOnce>().is_save = true;
                target_s22.GetComponent<saveOnce>().save_img(steps);
            }

        }
        steps++;

        Vector3 cur_loc = this.transform.localPosition;

        float forward_x = vectorAction[1] * speed;
        float forward_z = -vectorAction[0] * speed;

        Vector3 newposition1 = new Vector3(forward_x, 0, forward_z);
        //rBody.AddRelativeForce(newposition1 * speed);
        //rBody.MovePosition(mmTransform.localPosition + newposition1);
        //this.transform.Translate(newposition1, Space.Self);
        rBody.velocity = (newposition1);

        float distanceToTarget = Vector3.Distance(this.transform.position, target.transform.position);
        // Reached target

        if (distanceToTarget < 2.0f)
        {
            SetReward(10.0f);
            EndEpisode();
        }
        SetReward(-0.01f);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Obstacle"))
        {
            SetReward(-1.0f);
            //is_collide = true;
        }
        else
        {
            //is_collide = false;
        }
    }

    void OnCollisionStay(Collision collision)
    {
        if (collision.collider.CompareTag("Obstacle"))
        {
            SetReward(-1.0f);
            //is_collide = true;
        }
        else
        {
            //is_collide = false;
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
