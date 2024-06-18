#include <ros/ros.h>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Image.h>
#include <mutex>

double next_pulse_time, time_diff_gnss_local;
bool time_diff_valid = false, next_pulse_time_valid = false;

ros::Publisher pub_gps;

std::mutex m_time;

void gnss_tp_info_callback(const gnss_comm::GnssTimePulseInfoMsgConstPtr &tp_msg)
{
    gnss_comm::gtime_t tp_time = gnss_comm::gpst2time(tp_msg->time.week, tp_msg->time.tow);
    if (tp_msg->utc_based || tp_msg->time_sys == SYS_GLO)
        tp_time = gnss_comm::utc2gpst(tp_time);
    else if (tp_msg->time_sys == SYS_GAL)
        tp_time = gnss_comm::gst2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_BDS)
        tp_time = gnss_comm::bdt2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_NONE)
    {
        std::cerr << "Unknown time system in GNSSTimePulseInfoMsg.\n";
        return;
    }
    double gnss_ts = gnss_comm::time2sec(tp_time);

    std::lock_guard<std::mutex> lg(m_time);
    next_pulse_time = gnss_ts;
    next_pulse_time_valid = true;
}

void local_trigger_info_callback(const sensor_msgs::ImageConstPtr &msg) {
    std::lock_guard<std::mutex> lg(m_time);

    if (next_pulse_time_valid)
    {
        time_diff_gnss_local = next_pulse_time - msg->header.stamp.toSec();
        if (!time_diff_valid)       // just get calibrated
            ROS_INFO("time difference between GNSS and VI-Sensor got calibrated: %lf\n", time_diff_gnss_local);
        time_diff_valid = true;
        next_pulse_time_valid = false;
    }
}

void gps_callback(const sensor_msgs::NavSatFixConstPtr &msg) {
    if (!time_diff_valid)
        return;
    sensor_msgs::NavSatFix new_msg;
    new_msg.header.stamp.fromSec(msg->header.stamp.toSec() - time_diff_gnss_local);
    new_msg.status = msg->status;
    new_msg.latitude  = msg->latitude;
    new_msg.longitude = msg->longitude;
    new_msg.altitude  = msg->altitude;
    new_msg.position_covariance = msg->position_covariance;
    pub_gps.publish(new_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gps_sync_node");
    ros::NodeHandle n("~");

    pub_gps = n.advertise<sensor_msgs::NavSatFix>("/gps", 10);

    ros::Subscriber sub_gnss_time_pluse_info = n.subscribe("/ublox_driver/time_pulse_info", 100, gnss_tp_info_callback), 
                    sub_local_trigger_info = n.subscribe("/camera/infra1/image_rect_raw", 100, local_trigger_info_callback),
                    sub_gps = n.subscribe("/ublox_driver/receiver_lla", 100, gps_callback);
        
    ros::spin();

    return 0;
}
