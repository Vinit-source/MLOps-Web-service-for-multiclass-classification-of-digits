# ML Ops Repo

# Final Exam:

## Q1
## Q2
Show case use of docker. Running docker (with appropriate flags) should train and save models for variety of hyper-parameters. The models should be saved on host os and not just within the container. You need to make the changes to Dockerfile and/or docker commands. Submit your full docker commands to README.md, and attach a screenshot (refer to example image) that proves the models are accessible on host. Note the exact same timestamps of model creation (barring for the timezone diff of GMT+5.30)

Commands used:
```bash
$ sudo docker container ls

$ sudo docker run -v /media/vinitgore/Workplace/MTech/MTechYear2Sem1/MLOps/mnist_example/mnist_example/models:/exp/models -it mnist ls -l models
```

Screenshot of models on Docker:
![sc](images/Screenshot%20from%202021-11-27%2019-03-17.png)

Screenshot of models on host-os:
![hc](images/Screenshot%20from%202021-11-27%2019-03-29.png)

Complete Result:
```bash
(mlops) vinitgore@dell-Inspiron-15-3567:/media/vinitgore/Workplace/MTech/MTechYear2Sem1/MLOps/mnist_example$ sudo docker run -v /media/vinitgore/Workplace/MTech/MTechYear2Sem1/MLOps/mnist_example/mnist_example/models:/exp/models -it mnist ls -l /exp/models
total 0
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.25_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.25_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.5_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1.5_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_1_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.15_val_0.15_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.15_val_0.15_rescale_2.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.15_val_0.15_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.15_val_0.15_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.15_val_0.15_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_2_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_2_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_2_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_2_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:21 tt_0.15_val_0.15_rescale_2_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.25_val_0.25_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.25_val_0.25_rescale_0.25_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.25_val_0.25_rescale_0.25_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.25_val_0.25_rescale_0.25_gamma_0.1
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.25_val_0.25_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_0.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_0.5_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_1_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.25_val_0.25_rescale_2_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.25_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.25_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.5_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_1_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:22 tt_0.3_val_0.15_rescale_2_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.25_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.25_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.25_gamma_0.1
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.5_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:23 tt_0.3_val_0.3_rescale_1_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.3_val_0.3_rescale_2_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.25_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.25_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.25_gamma_0.1
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.5_gamma_0.01
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1.5_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1_gamma_0.001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_1_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2_gamma_0.0001
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2_gamma_1e-05
drwxrwxrwx 1 1000 1000 0 Nov 27 13:24 tt_0.4_val_0.2_rescale_2_gamma_1e-07
(mlops) vinitgore@dell-Inspiron-15-3567:/media/vinitgore/Workplace/MTech/MTechYear2Sem1/MLOps/mnist_example$ ls -l mnist_example/models
total 0
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.25_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.25_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.5_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1.5_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_1_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.15_val_0.15_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.15_val_0.15_rescale_2.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.15_val_0.15_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.15_val_0.15_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.15_val_0.15_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_2_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_2_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_2_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_2_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:51 tt_0.15_val_0.15_rescale_2_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.25_val_0.25_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.25_val_0.25_rescale_0.25_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.25_val_0.25_rescale_0.25_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.25_val_0.25_rescale_0.25_gamma_0.1
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.25_val_0.25_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_0.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_0.5_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_1_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.25_val_0.25_rescale_2_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.25_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.25_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.5_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_1_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:52 tt_0.3_val_0.15_rescale_2_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.25_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.25_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.25_gamma_0.1
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.5_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:53 tt_0.3_val_0.3_rescale_1_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.3_val_0.3_rescale_2_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.25_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.25_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.25_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.25_gamma_0.1
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.25_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.5_gamma_0.01
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_0.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1.5_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1_gamma_0.001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_1_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2.5_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2.5_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2.5_gamma_1e-06
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2.5_gamma_1e-07
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2_gamma_0.0001
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2_gamma_1e-05
drwxrwxrwx 1 vinitgore vinitgore 0 Nov 27 18:54 tt_0.4_val_0.2_rescale_2_gamma_1e-07
```