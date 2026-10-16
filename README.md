# Proactive Biometric Landmark Blinding Studio

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Computer Vision](https://img.shields.io/badge/Vision-MediaPipe%20Face%20Mesh-green.svg)](https://developers.google.com/mediapipe)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical research pipeline and interactive evaluation dashboard investigating targeted adversarial perturbations against automated facial landmark localization and canonical alignment in deepfake harvesting systems.

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Problem Statement & Core Hypothesis](#problem-statement--core-hypothesis)
- [How to Use the Application](#how-to-use-the-application)
- [Dashboard Views & Research Tabs](#dashboard-views--research-tabs)
- [System Architecture & Dataflow](#system-architecture--dataflow)
- [Mathematical Formulation & Metrics](#mathematical-formulation--metrics)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Installation & Getting Started](#installation--getting-started)
- [Operational Security & Relay Architecture](#operational-security--relay-architecture)
- [Academic Limitations & Research Scope](#academic-limitations--research-scope)
- [Contributing](#contributing)
- [License](#license)

---

## Executive Summary

Automated facial synthesis networks (such as deepfake generation pipelines and automated face-swappers) depend strictly on preliminary geometric alignment. These networks locate key facial landmarks (eyes, nose bridge, mouth corners) to solve an affine transformation matrix, rotating and scaling the face into a standardized canonical coordinate window ($112 \times 112$ px).

Rather than attempting post-hoc detection of synthesized media, this system implements a **proactive defense**: injecting imperceptible, localized spatial perturbations directly into key landmark regions. The objective is to distort the affine transformation matrix and trigger geometric alignment failures during automated scraping, all while preserving strict perceptual quality for human observers.

---

## Problem Statement & Core Hypothesis

### The Problem
Publicly shared facial images are vulnerable to automated biometric scraping and identity synthesis. If an attacker's automated ingestion pipeline can reliably extract landmark coordinates, the identity can be mapped onto arbitrary target geometries with high precision.

### The Research Hypothesis
> **Can subtle, localized spatial perturbations injected around key facial landmark anchors displace landmark detection and degrade canonical affine alignment beyond ordinary JPEG compression and uniform random noise, while maintaining high perceptual fidelity ($\text{PSNR} \ge 38\text{ dB}$, $\text{SSIM} \ge 0.96$)?**

The platform evaluates this hypothesis using normalized landmark drift, residual shifts in canonical alignment frames, and statistical hypothesis testing ($p$-values against stochastic noise baselines).

---

## How to Use the Application

### 1. Launching the App
Open PowerShell in the project directory and run:
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py