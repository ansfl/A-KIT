# A-KIT: Adaptive Kalman-Informed Transformer
Here, you can find the dataset and code of the deep learning architecture, **A-KIT**, which was introduced in the paper **A-KIT: Adaptive Kalman-Informed Transformer**.
## Abstract:
The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous underwater vehicle, we show that A-KIT outperforms the conventional EKF by more than 49.5\% and model-based adaptive EKF by an average of 35.4\% in terms of position accuracy.
<p float="left">
  <img src="https://github.com/ansfl/A-KIT/blob/main/Figs/path1.png" width="40%" />
  <img src="https://github.com/ansfl/A-KIT/blob/main/Figs/path2.png" width="40%" /> 
</p>

## Dataset:
We conducted experiments in the Mediterranean Sea near the shore of Haifa, Israel, using the Snapir AUV to gather data. Snapir is a modified, ECA Group, A18D mid-size AUV for deep-water applications. It performs autonomous missions up to 3000 meters in depth with 21 hours of endurance. Snapir is equipped with iXblue Phins Subsea, which is a FOG-based high-performance subsea inertial navigation system, and Teledyne RDI Work Horse navigator DVL that achieves accurate velocity measurements with a standard deviation of 0.02 [m/s]. The INS operates at a rate of 100 [Hz] and the DVL at 1[Hz].
The dataset was recorded on June 8, 2022, and contains approximately seven hours of data with different maneuvers, depths, and speeds. The train set is composed of eleven different data sections, each of a duration of 400 [sec] and with different dynamics for diversity. We used only 73.3 minutes of the recorded data for training. As ground truth (GT), we used the filter solution given by Delph INS, post-processing software for iXblue’s INS-based subsea navigation. To evaluate the approach, we examined an additional two 400 [sec] segments of the data that are not present in the training set, referring to them as the test set.
<p align="center">
  <img width="50%" height="50%" src="https://github.com/ansfl/A-KIT/blob/main/Figs/Snapir_AUV1.jpeg">
</p>

**Additional information regarding the dataset is located in the data folder.**
## Architecture and Algorithm
To cope with real-time adaptive process noise covariance matrix estimation, we propose A-KIT, an adaptive Kalman-informed transformer. To this end, we derive a tailored set-transformer network for time series data dedicated to real-time regression of the EKF’s process noise covariance matrix. Additionally, a Kalman-informed loss is designed to emulate the principles of the KF, enhancing the accuracy
of the process noise covariance. In this manner, A-KIT is designed as a hybrid algorithm combining the strengths of the well-established theory behind EKF and leveraging well-known, deep-learning characteristics. 
<p align="center">
  <img width="80%" height="80%" src="https://github.com/ansfl/A-KIT/blob/main/Figs/ProNet_arch.png">
</p>
<p align="center">
  <img width="50%" height="50%" src="https://github.com/ansfl/A-KIT/blob/main/Figs/flow chart.png">
</p>

# Citation

If you found our paper, code, or experimental data  useful for your research, please cite our paper:

  @article{cohen2025adaptive,
    title={Adaptive Kalman-Informed Transformer},
    author={Cohen, Nadav and Klein, Itzik},
    journal={Engineering Applications of Artificial Intelligence},
    volume={146},
    pages={110221},
    year={2025},
    publisher={Elsevier}
  }

