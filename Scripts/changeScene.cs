using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class changeScene : MonoBehaviour
{
    //private int num = 0;
    private GameObject root_env;
    private GameObject prev_env;
    public GameObject cur_env;
    private string[] all_env = new string[6] { "Env_1", "Env_2", "Env_3", "Env_4", "Env_5", "Env_6" };
    private int select_env = 1;
    private int old_env = 1;

    public bool change_scene = false;

    void Start()
    {
        root_env = GameObject.Find("Envs");
        cur_env = root_env.transform.Find(all_env[select_env]).gameObject;
        prev_env = root_env.transform.Find(all_env[old_env]).gameObject;

    }

    void Update()
    { 
        if (change_scene == true) 
        {
            select_env = UnityEngine.Random.Range(0, all_env.Length);
            while (select_env == old_env)
            {
                select_env = UnityEngine.Random.Range(0, all_env.Length);
            }
            cur_env = root_env.transform.Find(all_env[select_env]).gameObject;
            cur_env.SetActive(true);
            prev_env = root_env.transform.Find(all_env[old_env]).gameObject;
            prev_env.SetActive(false);
            old_env = select_env;
            change_scene = false;
            print(string.Format("change scene to {0}", select_env+1));
        }
    }

}
