
.. _stats_lmm_label:

Linear mixed models
===================

Linear mixed models (LMM) performs a linear regression analysis on residuals following a pre-determined distribution (commonly Gaussian). Thereby, it may quantify whether an independent factor (e.g. treatment yes/no) does effect a dependent factor (e.g. outcome). The relationship between input and output is defined via a formula, such as

.. math::
  x ~ a + b + a:b + (1|c)

Therein, x is the dependent factor, described by the independent factors a, b, and c. Of the independent factors, the factors :math:`a` and :math:`b` denote the impact of a & b whereas :math:`a:b` describes the interaction (conditional effect) between a and b. The latter refers to effects wherein e.g. the effect of a varies depending on the designation of b (note a:b is equal to b:a). Finally, :math:`(1|c)` defines a random effect.

Fixed & random effects
----------------------

Fixed effects are denoted as follows,

.. math::
  x ~ a

These assess whether there is *on average* a linear effect of a onto x. Fixed effects may be continuous or categorical. While continuous fixed effects add one degree of freedom per effect (e.g. :math:`a_1`, :math:`a_2`, :math:`a_3`, ...), categorical effects add one degree of freedom per expression level. Hence, if for example patient_id was added as a fixed effect, degrees of freedom equal to the number of patients would be added. This in turn significantly degrades statistical power.

While compound symmetry may required the addition of a *patient_id* to balance for repeated measurements, this issue may be solved without significantly inflating degree of freedom count (and thereby deflating statistical power). Namely, by re-classifying individual fixed factors as random factors. These are commonly (albeit not exclusively) categorical variables with several expression levels (such as *patient_id*). If modeled as a random factor, patient id's individual effects are approximated by a Gaussian distribution. Hence, instead of adding degrees of freedom equal to patient count, only two degrees of freedom are added (a mean and variance) per random factor. However, this comes at the drawback that the contribution of individual expression levels (e.g. patients) cannot be assessed anymore, as they are approximated by a distribution. Still, if variables are added merely to restore compound symmetry, it is unlikely that individual expression levels are of interest, making re-classification into random factors a highly attractive course of action.

Different types of random effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on the context, random effects can be a random intercept (e.g. different start value for different patients; :math:`(1|c)`), random slope (e.g. varying treatment effect size across patients; :math:`(a|c)`), or both (:math:`(1 + a|c)`). 

Normal, interaction, and nested effects
---------------------------------------

Most effects are modeled as fully independent factors :math:`a`. However individual factors may have different effects, depending on the expression of other factors. These relationship can be modeled in either of two ways, as interaction effects :math:`a:b` or nested effects :math:`a + a:b`. Therefor, if different expression levels exist only within individual expression levels of another factor, it is nested (e.g. patients are nested in hospitals). However, *just* some interaction between factors is expected without a strict hierarchy (e.g. age & sex), effects are modeled as interaction effects (:math:`a + b + a:b`)
