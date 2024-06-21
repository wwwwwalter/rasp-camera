# qt-plugins
export QT_QPA_PLATFORM_PLUGIN_PATH=/home/co2m/mambaforge/envs/opencv/lib/qt6/plugins/platforms
chmod u+x ./main_new
# start executable
while true; do
    ./main_new
    echo "return " $?
    echo "restarting..."
    sleep 1
done

