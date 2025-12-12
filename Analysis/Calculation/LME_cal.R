library(readxl)
library(nlme)
dat <- read_excel("Path to . xlsx data")

#
dat$res <- as.numeric(dat$res)

# 
dat$treatment_idx <- trimws(as.character(dat$treatment_idx))
dat$midx          <- trimws(as.character(dat$midx))

# 
dat$treatment_idx[
  grepl("^WT",  dat$treatment_idx, ignore.case = TRUE)] <- "WT"
dat$treatment_idx[
  grepl("^HOM$", dat$treatment_idx, ignore.case = TRUE)] <- "HOM"
dat$treatment_idx[
  grepl("HOMcon", dat$treatment_idx, ignore.case = TRUE)] <- "HOMcon"
dat$treatment_idx[
  grepl("HOMkv", dat$treatment_idx, ignore.case = TRUE)] <- "HOMkv"

# 
table(dat$treatment_idx, useNA = "ifany")

# 
dat$treatment_idx <- factor(dat$treatment_idx)
dat$treatment_idx <- relevel(dat$treatment_idx, ref = "WT")

#
dat$midx <- factor(dat$midx)

## 
dat_clean <- subset(dat,
                    !is.na(res) &
                      !is.na(treatment_idx) &
                      !is.na(midx))

cat(" =", nrow(dat),
    "； =", nrow(dat_clean), "\n")

## ==================
lme.obj <- lme(res ~ treatment_idx,
               random   = ~ 1 | midx,
               data     = dat_clean,
               method   = "ML",
               na.action = na.omit)

summary(lme.obj)