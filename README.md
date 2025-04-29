# Big Data Recommender System

This project implements a collaborative filtering recommender system for predicting missing user ratings.  
Developed for the Big Data course at USJ — ESIB.

---

## Overview

Given a user-item rating matrix with missing values, this system estimates the unknown ratings using:

- User-User Collaborative Filtering
- Item-Item Collaborative Filtering
- Global Baseline Estimate

The implementation is written in Python using NumPy and Pandas, with optional visualization via Seaborn and Matplotlib.

---

## Installation

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn
```

## Input Format

The input should be a CSV file with:

- Rows: users (identified by name or ID)
- Columns: items/products
- Cells: numerical ratings (1–5), with missing values left blank

Example like the file(`ratings.csv`):

```
     HP1,HP2,HP3,TW,SW1,SW2,SW3
A    4,5,1,,,,
B    5,5,4,,,,
C    2,4,5,,,,
D    3,3,,,,,

```
