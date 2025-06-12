This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to run and reproduce the results of [ExASPIM Flatfield Correction](https://codeocean.allenneuraldynamics.org/capsule/2321427/tree) on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://docs.codeocean.com/user-guide/compute-capsule-basics/managing-capsules/exporting-capsules-to-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)

# Instructions

## Download attached Data Assets

In order to fetch the Data Asset(s) this Capsule depends on, download them into the Capsule's `data` folder:
* [exaSPIM_flatfield_2024-02-19_12-47-17.tif](https://codeocean.allenneuraldynamics.org/data-assets/46324aa5-5d85-4a74-b17c-dab981ea00f9) should be downloaded to `data/flatfields`
* [updated_bkg_exaSPIM_667857_2024-02-19_12-47-17.tif](https://codeocean.allenneuraldynamics.org/data-assets/ab1ff52b-55f6-45ce-b70c-dbfa78b92511) should be downloaded to `data/bkg_images`
* [exaspim_template_round_4_brains](https://codeocean.allenneuraldynamics.org/data-assets/7d02cec9-1e13-4879-a4bc-24a9eab4d03c) should be downloaded to `data/exaspim_template_round_4_brains`
* [751473-masks-2025-03-18](https://codeocean.allenneuraldynamics.org/data-assets/606b62c0-e7af-4c84-95fe-9f34dd92d060) should be downloaded to `data/751473-masks-2025-03-18`
* [686951_masks](https://codeocean.allenneuraldynamics.org/data-assets/c66e349f-e048-44cb-b639-31db5c961cf0) should be downloaded to `data/686951_masks`

## Log in to the Docker registry

In your terminal, execute the following command, providing your password or API key when prompted for it:
```shell
docker login -u cameron.arshadi@alleninstitute.org registry.codeocean.allenneuraldynamics.org
```

## Run the Capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the Capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  --env AWS_ACCESS_KEY_ID=value \
  --env AWS_SECRET_ACCESS_KEY=value \
  --env AWS_DEFAULT_REGION=value \
  registry.codeocean.allenneuraldynamics.org/capsule/cc75cbb0-ded0-4224-8fa0-288967e590bf \
  bash run '' '' 0 gaussian False True True s3://aind-msma-morphology-data/test_data/exaSPIM_flatfield_2024-02-19_12-47-17.tif 1
```

As secrets are required, replace all `value` occurances with your personal credentials prior to your run.
