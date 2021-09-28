# use wget to download datasets

if [ $1 == 'm2cai' ]; then
  echo '==== Download into ./dataset/'$1' ===='
  mkdir -p './dataset/'$1
  wget -P './dataset/'$1 -c --http-user=m2cai_workflow --http-password=m2cai_workflow_challenge_2016 \
    http://camma.u-strasbg.fr/m2cai2016/datasets/workflow/m2cai16-workflow.zip
elif [ $1 == 'cholec80' ]; then
  echo '==== Download into ./dataset/'$1' ===='
  mkdir -p './dataset/'$1
  wget -P './dataset/'$1 -c --http-user=camma_cholec80 --http-password=cholec80_unistra \
    http://camma.u-strasbg.fr/datasets/cholec80/cholec80.zip
  mkdir -p './dataset/'$1
else
  echo 'Invalid Dataset'
fi
