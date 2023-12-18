# CYBERINSIGHT - Total Cost of Ownership Analysis for High-Performance Computing workloads in the Cloud

The research computing landscape -- hardware, software, applications, and expectations -- continues to evolve rapidly. Machine learning and big data analytics are creating new opportunities to extract research insight from experimental and computational data. Concurrently, cloud service providers are rapidly deploying new computing and data analysis services atop hardware configurations that are increasingly similar to those found in academic environments. All of this ferment is convolved with a new paradigm of streaming data from a world of scientific instruments and ubiquitous wireless sensors. Concurrently, new expectations for data management and retention, combined with resource constraints, are putting new pressures on research agenciesand academic institutions to find cost-effective approaches that maximize scientific benefit while minimizing costs.

This project proposes to build, publish, and regularly update comparisons of the evolving total cost of ownership (TCO) of on-premise HPC clusters and data storage systems relative to NSF-supported facilities and commercial cloud services such as Amazon AWS, Microsoft Azure, and Google Cloud Platform. Concretely, the team proposes to build and host a Jupyter notebook that enables the cyberinfrastructure community to add, modify, and explore total cost of ownership models based on a variety of usage patterns and performance expectations. This will facilitate community building, shared experimentation and comparisons.

## Public Cloud

**Public cloud** refers to companies offering hosted or managed computing services to a general-purpose *public customer base*. These companies are virtually always private entities themselves.

## Private cloud

**Private cloud** or **On-prem** infrastructure refers to computational services that are maintained and managed to support a general or specialized *private user base*, often affiliated researchers. Traditional HPC centers are on-premise and thus employ a diverse technical staff to manage compute clusters, storage, networking, and other aspects of high-performance computing (HPC).

## Total cost of ownership

Total cost of ownership (TCO) consists of capital expenditures (CapEx) and operational expenditures (OpEx). CapEx includes purchases of computing infrastructure; large, infrequent purchases that yield value over a multi-year period. OpEx includes power costs and service fees, such as paying for compute cycles on a managed server. Significant CapEx is required to set up a modern HPC cluster with current-gen CPUs, memory, accelerators, storage, and network. Subscribing to a public cloud provider requires no CapEx beyond a web-browsing device connected to the Internet. However, it is not immediately obvious if the *total cost of ownership* of a HPC datacenter plus power, floorspace, maintenance costs is more or less expensive over ~5 years than running the same workloads on public cloud provider instances of similar quality. This project aims to provide a framework for addressing that comparison.

## Prerequisites:

These requirements are listed in `requirements.txt`.

* Python 3.7+
* `jupyter`
* `matplotlib`
* `ipyfilechooser`

## Installing prerequisites

### Option A - Install with `conda` (Recommended)

1. Ensure `conda` is installed in some form.
  * https://docs.conda.io/en/latest/miniconda.html

2. `conda create -n hpc-tco python=3.7 jupyter matplotlib ipyfilechooser`


### Option B - Install into system Python with `pip`

1. Ensure `python3 --version` is 3.7+
  * https://www.python.org/downloads/
  * Your package manager may also include python 3.7+

2. `python3 -m pip install -r requirements.txt`
  * You may optionally install these packages inside a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).


## Launching the notebook server

1. Ensure we are in a valid python environment 
  * (Conda: `conda activate hpc-tco` after following the `conda` installation steps)

2. Clone this repository. 

3. Run `jupyter notebook` from inside the cloned directory (i.e. where this README.md is located).
  * See `jupyter notebook --help` for additional options.
