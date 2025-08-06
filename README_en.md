English | [中文](README_zh.md)

# MindCos

- [Introduction to MindCos](#introduction-to-mindcos)
- [Application Scenarios](#application-scenarios)
- [Installation Guide](#installation-guide)

## Introduction to MindCos

Cosmology explores the fundamental nature, origin, and evolution of the universe, relying on complex experimental data from telescopes, particle accelerators, and other advanced instruments. These experiments generate vast and intricate datasets that challenge traditional data processing methods. To address this, MindCos leverages advanced deep learning techniques to enhance the efficiency and accuracy of data analysis in cosmology.

MindCos, built on the MindSpore framework, is an AI-powered toolkit designed for cosmological research. It supports a variety of applications, including jet identification, magnetic field prediction, and solving key equations like the Grad–Shafranov, Helmholtz, and Lane–Emden equations. These applications are critical in cosmology:
- **Jet Identification**: Analyzes particle jets from high-energy collisions to study quark-gluon plasma and cosmic events, aiding in understanding early universe conditions.
- **Magnetic Field Prediction**: Uses Physics-Informed Neural Networks (PINNs) to model magnetic fields in cosmic structures, collider, or accelerator, solving Maxwell’s equations to predict the magnetic fields.
- **Grad–Shafranov Equation**: Solves plasma equilibrium in astrophysical contexts, such as stellar magnetic fields in stars or accretion disks around compact objects like black holes, critical for modeling magnetohydrodynamic processes in cosmological systems.
- **Helmholtz Equation**: Models wave propagation in cosmological simulations, relevant for studying gravitational waves or cosmic microwave background fluctuations, using synthetic boundary and collocation data in a rectangular domain.
- **Lane–Emden Equation**: Describes the structure of polytropic stars, crucial for modeling stellar evolution, star formation, and galactic dynamics in cosmological studies.

MindCos significantly reduces computational costs and processing time while improving accuracy compared to traditional methods. It is a versatile and user-friendly tool for researchers, professors, and students, empowering them to explore the universe's mysteries with greater precision.

## Application Scenarios

The following table summarizes the supported application scenarios, datasets, model architectures, and hardware compatibility:

| Application Scenario | Dataset | Model Architecture | CPU | Ascend |
|---------------------|---------|--------------------|-----|--------|
| Grad–Shafranov Equation (Stellar Magnetic Field) | Synthetic boundary points on a D-shaped contour and collocation points for plasma equilibrium in stellar magnetic field simulations | PINN | ✔️ | ✔️ |
| Grad–Shafranov Equation (Accretion Disk) | Synthetic boundary points on a drop-shaped contour and collocation points for plasma equilibrium in accretion disk simulations | PINN | ✔️ | ✔️ |
| Helmholtz Equation | Synthetic boundary and collocation points in a rectangular domain for wave propagation simulations | PINN | ✔️ | ✔️ |
| Lane–Emden Equation | Synthetic boundary and collocation points in a Cartesian domain for polytropic stellar structure simulations | PINN | ✔️ | ✔️ |
| Jet Identification | Quark-gluon jet dataset | LorentzNet | ✔️ | ✔️ |
| Magnetic Field Prediction | Synthetic 3D spatial points with magnetic field components for cosmic, collider or accelerator magnetic field simulations | PINN | ✔️ | ✔️ |

## Installation Guide

Follow the instructions below to install MindCos and its dependencies.

### Install MindSpore
```bash
# Refer to the official MindSpore installation guide
https://www.mindspore.cn/install
```

Example installation command:
```bash
conda create -n MindCos python==3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/x86_64/mindspore-2.2.14-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install GPU Version of MindSpore
For Ascend or GPU support, refer to the [installation instructions](gpu_version_install.txt).