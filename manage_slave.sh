#!/bin/bash
#
# Normally there would be a conf/slaves file but Spark root is not modifiable
# by regular users so we'll start all by hand here. Also need to set a local
# working dir that can be accessed by a normal user.


export SPARK_LOCAL_DIRS=$HOME/spark/tmp
export SPARK_WORKER_DIR=$SPARK_LOCAL_DIRS
export SPARK_LOG_DIR=$SPARK_LOCAL_DIRS
mkdir -p $SPARK_LOCAL_DIRS

case $1 in

    start)
#        /opt/spark/2.3.0/sbin/start-slave.sh spark://compute-0-0:7077
        /opt/spark/2.3.0/sbin/start-slave.sh spark://pc-171.calit2.optiputer.net:7077
#        /opt/spark/2.3.0/sbin/start-slave.sh spark://10.1.171.1:7077
        ;;
    stop)
        /opt/spark/2.3.0/sbin/stop-slave.sh
        ;;
    *)
        echo "Usage: manage_*.sh [start|stop]"
        ;;
esac
