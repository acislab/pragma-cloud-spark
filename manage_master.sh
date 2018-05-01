#!/bin/bash
#
# Normally there would be a conf/slaves file but Spark root is not modifiable
# by regular users so we'll start all by hand here. Also need to set a local
# working dir that can be accessed by a normal user.


export SPARK_LOCAL_DIRS=$HOME/spark/tmp
export SPARK_WORKER_DIR=$SPARK_LOCAL_DIRS
export SPARK_LOG_DIR=$SPARK_LOCAL_DIRS
mkdir -p $SPARK_LOCAL_DIRS

#export SPARK_NO_DAEMONIZE=true

case $1 in

    start)
        /opt/spark/2.3.0/sbin/start-master.sh
        ;;
    stop)
        /opt/spark/2.3.0/sbin/stop-master.sh
        ;;
    *)
        echo "Usage: manage_*.sh [start|stop]"
        ;;
esac
