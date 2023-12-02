# File type is .py
# Table states-schema, columns = [data(STRING)]


import requests
import json, os
import tempfile
import time

def upload_file(
    stream,
    data,
    api_key="G4566a9afaR3e2U_974s5a23fl8P0_7ydb3i0dNd9s7iNek6f0e7d6aaRea0fWsecb7e9yE6Q3Mmhb2aaaQe3McMp3G3b9c334qiWaCdSb9",
    integration_id="653c4e6450ab21a8e14676cc",
    
):

    fPath = tempfile.mkstemp(suffix='.json')[1]
    open(fPath, 'w').write(json.dumps(data))
    print(fPath)
    data_file = open(fPath, 'rb')

    response = requests.post(
        url=f"https://api.v2.datagran.io/v2/integrations/{integration_id}/push_data/{stream}",
        headers={"Authorization": api_key},
        files={'data-file': data_file}
    )
    print(f"Upload: The response HTTP status code is: {response.status_code}")
    print(response.content)
    if response.status_code == 200:
        task_id = response.json()["bg_task_id"]
        for i in range(1):
            response = requests.get(
                url=f"https://api.v2.datagran.io/v2/integrations/{integration_id}/push_data_status/{task_id}",
                headers={"Authorization": api_key},
            )
            print(f"Followup ({task_id}): The response HTTP status code is: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                if not isinstance(result, dict):
                    print(f"Followup ({task_id}): Unexpected content format! Aborting.")
                    return
                if '_error' in result:
                    print(f"Followup ({task_id}): Unexpected error while checking status: {result['_error']}. Aborting.")
                    return
                if 'status' in result:
                    status = result['status']
                    if status in ['CANCELLED', 'ERROR', 'SUCCESSFUL']:
                        print(f"Followup ({task_id}): The status is: " + status)
                        print(f"Followup ({task_id}): The results are: {result.get('output', {})}")
                        return
                    else:
                        time.sleep(1)
                else:
                    print(f"Followup ({task_id}): Unexpected content format! Aborting.")
                    return
            else:
                print(f"Followup ({task_id}): Unexpected HTTP status code! Aborting.")
                return
        print(f"Followup ({task_id}): Timed Out")