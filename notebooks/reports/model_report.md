# Model Report

Here we make comparisons across models; the style of this document is that each section answers/frames a question to be used/discussed in the paper.

## Can we use autoencoders for image reconstruction?

- Look at the baseline model; yes the autoencoders appear to be learning the mapping.
- This model shows the barebones capability, without any fancy training or probabilistic treatment.
- The numbers correspond to the mean pixelwise cross-entropy; larger is worse.

![baseline-best](outputs/baseline/baseline_bestimgs.png)

- In the worst performing in terms of loss, we don't do too badly still, as the images look pretty good
  - Losses are larger because there are more bright pixels

![baseline-worst](outputs/baseline/baseline_worstimgs.png)

- In terms of the radial distribution, we do a pretty good job and noise-free too.
- Comparing with one conventional method, onion peeling, we don't have a noisy baseline to deal with.

![baseline-radial](outputs/baseline/baseline_common_radial.png)

