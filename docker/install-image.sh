#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
PKG_DIR=$(dirname ${RUN_DIR})

SRC_IMAGE=""

function usage_exit {
  cat <<_EOS_ 1>&2
  Usage: install-env-docker.sh [OPTIONS...]
  OPTIONS:
    -h, --help                      このヘルプを表示
    -i, --image DOCKER_IMAGE[:TAG]  ベースとなるDockerイメージを指定
_EOS_
  exit 1
}

while (( $# > 0 )); do
  if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    usage_exit
  elif [[ $1 == "-i" ]] || [[ $1 == "--image" ]]; then
    if [[ $2 == -* ]]; then
      echo "無効なパラメータ"
      usage_exit
    else
      SRC_IMAGE=$2
    fi
    shift 2
  else
    echo "無効なパラメータ: $1"
    usage_exit
  fi
done

if [[ -z ${SRC_IMAGE} ]]; then
  echo "ベースとなるDockerイメージを指定してください"
  usage_exit
fi

SRC_IMAGE_ARR=(${SRC_IMAGE//:/ })

IMAGE_EXIST=$(docker images ${SRC_IMAGE} | grep ${SRC_IMAGE_ARR[0]})
if [[ -z ${IMAGE_EXIST} ]]; then
  echo "指定したDockerイメージは存在しません"
  usage_exit
fi

docker build \
  -t "${SRC_IMAGE}-h5" \
  --build-arg SRC_IMAGE="${SRC_IMAGE}" \
  ${RUN_DIR}

if [[ $? != 0 ]]; then
    echo "エラーにより中断しました．"
    exit 1
fi