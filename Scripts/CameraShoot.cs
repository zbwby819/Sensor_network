using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System;

public class CameraShoot : MonoBehaviour
{
    public Camera sensorCam;
    public Camera sensorCam2;
    public Camera sensorCam3;
    public Camera sensorCam4;

    public GameObject _target;
    public String sensor_name;
    //private bool start = false;

    public float range_x_min=0.0f;
    public float range_x_max=39.0f;
    public float loc_y = 0.5f;
    public float range_z_min=-39.0f;
    public float range_z_max=0.0f;
    public float obstacleCheckRadius = 2.0f;

    public GameObject target_s2;
    public GameObject target_s3;
    public GameObject target_s4;
    public GameObject target_s5;
    public GameObject target_s6;
    public GameObject target_s7;
    public GameObject target_s8;
    public GameObject target_s9;
    public GameObject target_s10;
    public GameObject target_s11;
    public GameObject target_s12;

    public GameObject random_s1;

    bool c_flag = false;
    public int num = 1;
    Texture2D CaptureCamera(Camera camera, Camera camera2, Camera camera3, Camera camera4, Rect rect)
    {
        RenderTexture rt = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera.targetTexture = rt;
        camera.Render();
        RenderTexture.active = rt;
        Texture2D screenShot = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot.ReadPixels(rect, 0, 0);
        screenShot.Apply();
        byte[] bytes = screenShot.EncodeToPNG();
        string filename = "C:/Sensor_network/training/" + sensor_name + "/1/" + num + ".png";
        System.IO.File.WriteAllBytes(filename, bytes);
        //Debug.Log(string.Format("Camera is saved: {0}", filename));
        camera.targetTexture = null;
        RenderTexture.active = null; 
        GameObject.Destroy(rt);

        RenderTexture rt2 = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera2.targetTexture = rt2;
        camera2.Render();
        RenderTexture.active = rt2;
        Texture2D screenShot2 = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot2.ReadPixels(rect, 0, 0);
        screenShot2.Apply();
        byte[] bytes2 = screenShot2.EncodeToPNG();
        string filename2 = "C:/Sensor_network/training/" + sensor_name + "/2/" + num + ".png";
        System.IO.File.WriteAllBytes(filename2, bytes2);
        //Debug.Log(string.Format("Camera is saved: {0}", filename2));
        camera2.targetTexture = null;
        RenderTexture.active = null;
        GameObject.Destroy(rt2);

        RenderTexture rt3 = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera3.targetTexture = rt3;
        camera3.Render();
        RenderTexture.active = rt3;
        Texture2D screenShot3 = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot3.ReadPixels(rect, 0, 0);
        screenShot3.Apply();
        byte[] bytes3 = screenShot3.EncodeToPNG();
        string filename3 = "C:/Sensor_network/training/" + sensor_name + "/3/" + num + ".png";
        System.IO.File.WriteAllBytes(filename3, bytes3);
        //Debug.Log(string.Format("Camera is saved: {0}", filename3));
        camera3.targetTexture = null;
        RenderTexture.active = null;
        GameObject.Destroy(rt3);

        RenderTexture rt4 = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera4.targetTexture = rt4;
        camera4.Render();
        RenderTexture.active = rt4;
        Texture2D screenShot4 = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot4.ReadPixels(rect, 0, 0);
        screenShot4.Apply();
        byte[] bytes4 = screenShot4.EncodeToPNG();
        string filename4 = "C:/Sensor_network/training/" + sensor_name + "/4/" + num + ".png";
        System.IO.File.WriteAllBytes(filename4, bytes4);
        //Debug.Log(string.Format("Camera is saved: {0}", filename4));
        camera4.targetTexture = null;
        RenderTexture.active = null;
        GameObject.Destroy(rt4);

        return screenShot;
    }

    private bool IsInView(Vector3 _target_pos)
    {
        Transform camTransform = Camera.main.transform;
        Vector2 viewPos = Camera.main.WorldToViewportPoint(_target_pos);
        Vector3 dir = (_target_pos - camTransform.position).normalized;
        float dot = Vector3.Dot(camTransform.forward, dir);
        if (dot > 0 && viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1)
            return true;
        else
            return false;
    }

    public void Save(Vector3 _target_pos)
    {
        //FileStream fs = new FileStream(Application.dataPath + "/target_loc.txt", FileMode.Append);
        FileStream fs = new FileStream("C:/Sensor_network/target_loc.txt", FileMode.Append);
        byte[] bytes = new UTF8Encoding().GetBytes(_target_pos.ToString() + string.Format("{0}\r\n", num));
        fs.Write(bytes, 0, bytes.Length);
        fs.Close();

    }

    void random_target()
    {
        c_flag = true;
        float ni = UnityEngine.Random.Range(range_x_min, range_x_max);
        float nt = UnityEngine.Random.Range(range_z_min, range_z_max);
    
        while (c_flag == true)
        {
            c_flag = false;
            Vector3 position1 = new Vector3(ni, 0.75f, nt);
            Collider[] colliders1 = Physics.OverlapSphere(position1, obstacleCheckRadius);
            foreach (Collider col in colliders1)
            {
                // If this collider is tagged "Obstacle"
                if (col.tag == "Obstacle")
                {
                    // Then this position is not a valid spawn position
                    c_flag = true;
                    ni = UnityEngine.Random.Range(range_x_min, range_x_max);
                    nt = UnityEngine.Random.Range(range_z_min, range_z_max);
                }
            }

        }
        _target.transform.position = new Vector3(ni, 0.75f, nt);
    }

    void Update()
    {

        CaptureCamera(sensorCam, sensorCam2, sensorCam3, sensorCam4, new Rect(0, 0, 128, 72));
        target_s2.GetComponent<saveSensor_2>().num = num;
        target_s2.GetComponent<saveSensor_2>().camsave(num);
        target_s3.GetComponent<saveSensor_2>().num = num;
        target_s3.GetComponent<saveSensor_2>().camsave(num);
        target_s4.GetComponent<saveSensor_2>().num = num;
        target_s4.GetComponent<saveSensor_2>().camsave(num);
        target_s5.GetComponent<saveSensor_2>().num = num;
        target_s5.GetComponent<saveSensor_2>().camsave(num);
        target_s6.GetComponent<saveSensor_2>().num = num;
        target_s6.GetComponent<saveSensor_2>().camsave(num);
        target_s7.GetComponent<saveSensor_2>().num = num;
        target_s7.GetComponent<saveSensor_2>().camsave(num);
        target_s8.GetComponent<saveSensor_2>().num = num;
        target_s8.GetComponent<saveSensor_2>().camsave(num);
        target_s9.GetComponent<saveSensor_2>().num = num;
        target_s9.GetComponent<saveSensor_2>().camsave(num);
        if(target_s10)
        {
            target_s10.GetComponent<saveSensor_2>().num = num;
            target_s10.GetComponent<saveSensor_2>().camsave(num);
        }
        if (target_s11)
        {
            target_s11.GetComponent<saveSensor_2>().num = num;
            target_s11.GetComponent<saveSensor_2>().camsave(num);
        }
        if (target_s12)
        {
            target_s12.GetComponent<saveSensor_2>().num = num;
            target_s12.GetComponent<saveSensor_2>().camsave(num);
        }

        if (random_s1)
        {
            random_s1.GetComponent<randomSensor_2>().num = num;
            random_s1.GetComponent<randomSensor_2>().camsave(num);
            random_s1.GetComponent<randomSensor_2>().locsave(num);
            random_s1.GetComponent<randomSensor_2>().randomloc();
        }

        Vector3 _target_pos = _target.transform.position;
        Save(_target.transform.localPosition);
        //print(_target.transform.localPosition);
        random_target();
        num++;
    }
}
