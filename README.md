# ML Ops Repo

# Assignment 10:

1. Serve SVM and Decision tree models using the flask on separate relative URLs. i.e. `localhost:8000/svm_predict` and `localhost:8000/decision_tree_predict`. (ip could be different than localhost, in your case)
2. Dockerize the deployment i.e. create dockerfile and build image such that when you do `docker run` (may be with some more flags), the above two links should be accessible via curl. Write `docker_example.sh` shell script that includes the full curl commands.
