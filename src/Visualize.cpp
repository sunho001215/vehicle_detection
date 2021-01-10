#include "Visualize.h"


void DrawRotatedRectangle(cv::Mat& image, cv::Point centerPoint, cv::Size rectangleSize, double rotationDegrees)
{
    cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

    // Create the rotated rectangle
    cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    cv::Point vertices_temp;
    vector<cv::Point> vertices;    
    for(int i = 0; i < 4; ++i){
        vertices_temp = vertices2f[i];
        vertices.push_back(vertices_temp);
    }

    // Now we can fill the rotated rectangle with our specified color
    cv::polylines(image, vertices, true, Scalar(255, 255, 255), 2, LINE_AA);

    vertices.clear();
    vector<cv::Point>(vertices).swap(vertices);
}