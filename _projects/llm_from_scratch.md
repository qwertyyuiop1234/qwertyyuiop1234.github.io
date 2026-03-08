---
layout: page
title: LLM from scratch [Ongoing]
description: a project building an LLM from scratch in python
img: assets/img/llm_from_scratch.png
importance: 1
category: work
---

This project is a personal journey to build a Large Language Model (LLM) completely from scratch. 

## Overview
Inspired by and referencing the tutorial [LLMs from Scratch – Practical Engineering from Base Model to PPO RLHF](https://www.youtube.com/watch?v=p3sij8QzONQ&t=1304s), I am implementing the core components of the LLM architecture step-by-step. The goal is to gain a deep, practical understanding of how modern language models process text, learn patterns, and generate responses by coding everything **myself** rather than relying on high-level abstractions.

## Core Objectives
- Construct the Transformer architecture (Attention, Encoders/Decoders) from the ground up.
- Understand tokenization, positional encoding, and self-attention algorithms mathematically and programmatically.
- Maintain minimal external dependencies to ensure a fundamental grasp of tensor operations.

## Devlog

<ul>
{% for post in site.posts %}
  {% if post.tags contains 'llm-project' %}
    <li>{{ post.date | date: "%Y-%m-%d" }} : <a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% endif %}
{% endfor %}
</ul>
