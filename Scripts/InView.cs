using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System;


public class InView : MonoBehaviour
{
    
    int num = 1;
    public GameObject _target;
    Transform _target_tf;
    Vector3 _target_pos;

    private void Update()
    {
        _target_tf = _target.transform;
        _target_pos = _target_tf.position;
        
        if (IsInView(_target_pos))
        {
            Save(_target_pos);
        }
        num++;
    }

    private bool IsInView(Vector3 _target_pos)
    {
        Transform camTransform = Camera.main.transform;
        Vector2 viewPos = Camera.main.WorldToViewportPoint(_target_pos);
        Vector3 dir = (_target_pos - camTransform.position).normalized;
        float dot = Vector3.Dot(camTransform.forward, dir);//判断物体是否在相机前面
        if (dot > 0 && viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1)
            return true;
        else
            return false;
    }
    public void Save(Vector3 _target_pos)
    {
        //写文件 文件名为save.text
        //这里的FileMode.create是创建这个文件,如果文件名存在则覆盖重新创建
        FileStream fs = new FileStream(Application.dataPath + "/target_loc.txt", FileMode.Append);
        //存储时时二进制,所以这里需要把我们的字符串转成二进制
        byte[] bytes = new UTF8Encoding().GetBytes(_target_pos.ToString() + string.Format("{0}  ", num));
        fs.Write(bytes, 0, bytes.Length);
        Debug.Log(string.Format("保存坐标:{0}", num ));
        //每次读取文件后都要记得关闭文件
        fs.Close();
    }

}