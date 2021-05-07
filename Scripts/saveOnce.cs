
using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System;

public class saveOnce : MonoBehaviour
{
    public Camera sensorCam;
    public Camera sensorCam2;
    public Camera sensorCam3;
    public Camera sensorCam4;

    public String sensor_name;
    public bool is_save = true;
    private int num = 1;
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
        string filename = "C:/Cambridge/Sensor_network/training/" + sensor_name + "/1/" + num + ".png";
        System.IO.File.WriteAllBytes(filename, bytes);
        //Debug.Log(string.Format("Camera is saved: {0}", filename));
        camera.targetTexture = null;
        RenderTexture.active = null;
        //GameObject.Destroy(rt);

        RenderTexture rt2 = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera2.targetTexture = rt2;
        camera2.Render();
        RenderTexture.active = rt2;
        Texture2D screenShot2 = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot2.ReadPixels(rect, 0, 0);
        screenShot2.Apply();
        byte[] bytes2 = screenShot2.EncodeToPNG();
        string filename2 = "C:/Cambridge/Sensor_network/training/" + sensor_name + "/2/" + num + ".png";
        System.IO.File.WriteAllBytes(filename2, bytes2);
        //Debug.Log(string.Format("Camera is saved: {0}", filename2));
        camera2.targetTexture = null;
        RenderTexture.active = null;
        //GameObject.Destroy(rt2);

        RenderTexture rt3 = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera3.targetTexture = rt3;
        camera3.Render();
        RenderTexture.active = rt3;
        Texture2D screenShot3 = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot3.ReadPixels(rect, 0, 0);
        screenShot3.Apply();
        byte[] bytes3 = screenShot3.EncodeToPNG();
        string filename3 = "C:/Cambridge/Sensor_network/training/" + sensor_name + "/3/" + num + ".png";
        System.IO.File.WriteAllBytes(filename3, bytes3);
        //Debug.Log(string.Format("Camera is saved: {0}", filename3));
        camera3.targetTexture = null;
        RenderTexture.active = null;
        //GameObject.Destroy(rt3);

        RenderTexture rt4 = new RenderTexture((int)rect.width, (int)rect.height, -1);
        camera4.targetTexture = rt4;
        camera4.Render();
        RenderTexture.active = rt4;
        Texture2D screenShot4 = new Texture2D((int)rect.width, (int)rect.height, TextureFormat.RGB24, false);
        screenShot4.ReadPixels(rect, 0, 0);
        screenShot4.Apply();
        byte[] bytes4 = screenShot4.EncodeToPNG();
        string filename4 = "C:/Cambridge/Sensor_network/training/" + sensor_name + "/4/" + num + ".png";
        System.IO.File.WriteAllBytes(filename4, bytes4);
        //Debug.Log(string.Format("Camera is saved: {0}", filename4));
        camera4.targetTexture = null;
        RenderTexture.active = null;
        //GameObject.Destroy(rt4);

        return screenShot;
    }

    public void save_img(int img_num=1)
    {
        if (is_save)
        {
            num = img_num;
            CaptureCamera(sensorCam, sensorCam2, sensorCam3, sensorCam4, new Rect(0, 0, 128, 72));
            //print(("Camera is saved:{0}", this.name));
            is_save = false;
            //num++;
        }
    }

    void Start()
    {
        print($"Camera is saved:{this.name}");
        save_img(1);
    }

}
