using System;
using UnityEngine;
using System.Collections;
using System.IO;
using System.Text;
using System.CodeDom;

public class testRead : MonoBehaviour
{
    public GameObject target;
    public GameObject robot;

    public void Save()
    {
        Vector3 target_pos = target.transform.localPosition;
        Vector3 robot_pos = robot.transform.localPosition;
        byte[] bytes = new UTF8Encoding().GetBytes(target_pos.ToString() + string.Format("\n"));
        byte[] bytes2 = new UTF8Encoding().GetBytes(robot_pos.ToString() + string.Format("\n"));
        FileStream fs = new FileStream("C:/Sensor_network/zz/target_loc.txt", FileMode.Append);
        FileStream fs2 = new FileStream("C:/Sensor_network/zz/robot_loc.txt", FileMode.Append);
        fs.Write(bytes, 0, bytes.Length);
        fs2.Write(bytes2, 0, bytes2.Length);
        //每次读取文件后都要记得关闭文件
        fs.Close();
        fs2.Close();
    }

    //读取
    public Tuple<float[], float[], float[], float[]> Loadloc()
    {
        int f_length = 2000;
        //FileMode.Open打开路径下的save.text文件
        FileStream fs = new FileStream("C:/Sensor_network/zz/target_loc.txt", FileMode.Open);
     
        byte[] bytes = new byte[f_length];
        fs.Read(bytes, 0, f_length);
        string s = new UTF8Encoding().GetString(bytes);

        FileStream fs2 = new FileStream("C:/Sensor_network/zz/robot_loc.txt", FileMode.Open);
        byte[] bytes2 = new byte[f_length];
        //fs2.Read(bytes2, 0, fs2.Length);
        fs2.Read(bytes2, 0, f_length);
        string s2 = new UTF8Encoding().GetString(bytes2);

        string[] t_loc_string = s.Split('\n');
        string[] r_loc_string = s2.Split('\n');

        float[] t_loc_x = new float[100];
        float[] t_loc_y = new float[100];
        float[] r_loc_x = new float[100];
        float[] r_loc_y = new float[100];

        for (int i = 0; i < t_loc_string.Length; i++)
        {
            //Debug.Log(t_loc_string[i]);
            string[] sArray = t_loc_string[i].Split(new char[3] {'(', ',', ')'});
            //Debug.Log(sArray[1]);
            t_loc_x[i] = float.Parse(sArray[1]);
            t_loc_y[i] = float.Parse(sArray[3]);
            //Debug.Log(t_loc_x[i]);
            //Debug.Log(t_loc_y[i]);
            string[] sArray2 = r_loc_string[i].Split(new char[3] { '(', ',', ')' });
            //Debug.Log(r_loc_string[i]);
            //Debug.Log(sArray[2]);
            r_loc_x[i] = float.Parse(sArray2[1]);
            r_loc_y[i] = float.Parse(sArray2[3]);
        }
        Tuple<float[], float[], float[], float[]> all_loc = new Tuple<float[], float[], float[], float[]>(t_loc_x, t_loc_y, r_loc_x, r_loc_y);
        return all_loc;
    }

    void Start()
    {
        //Save();
        //(float[] t_locx, float[] t_locy, float[] r_locx, float[] r_locy) =Loadloc();
        //Debug.Log(t_locx[0]);
        //Debug.Log(t_locy[100]);
        //Debug.Log(r_locx[20]);
        //Debug.Log(r_locy[99]);

    }
}
