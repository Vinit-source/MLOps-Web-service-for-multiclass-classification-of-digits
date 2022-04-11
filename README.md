# Web service for mutliclass digit classification

## Scope

**1.** Serve SVM and Decision tree models using the flask on separate relative URLs. i.e. `localhost:8000/svm_predict` and `localhost:8000/decision_tree_predict`. (ip could be different than localhost, in your case)     
**2.** Dockerize the deployment i.e. create dockerfile and build image such that when you do `docker run` (may be with some more flags), the above two links should be accessible via curl. Write `docker_example.sh` shell script that includes the full curl commands.

## Results:

### Client:
![client](images/Screenshot%20from%202021-11-29%2004-10-02.png)

### Server:
![server](images/Screenshot%20from%202021-11-29%2004-09-52.png)

**3.** Deploy on Google Kubernetes Engine    
The `deployment.yaml` file was used as config file for deployment.

Tasks performed during deployment:
* Authenticate Docker on cloud shell.
* Push docker image of the web application to Artifact Registry.
* Create a standard cluster of 3 nodes.
* Authorize cluster to create its entry for in kubeconfig.
* Deploy app image using deployment.yaml and check status.
* Expose deployment on Kubernetes service to public.
* Make requests to running deployment using curl API calls on the EXTERNAL-IP provided by the service.
* Perform manual scaling.
* Create a HorizontalPodAutoScaler for auto-scaling.
* Resize cluster size by changing number of nodes.
* Update app, create image and set the deployment with the new image (version).
* Check background workings like forming of ReplicaSets and termination of Pods.

The results and experiments performed during deployment can be found in [`kubernetes/deployment_results.txt`](kubernetes/deployment_results.txt)   

Specifically:    
![result](kubernetes/Screenshot%20from%202022-04-11%2021-32-27.png)

Accessible from external-IP:    
![deployed](kubernetes/Screenshot%20from%202022-04-11%2017-04-07.png)


