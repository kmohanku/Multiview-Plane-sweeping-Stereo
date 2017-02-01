#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <tuple>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

Mat intrinsic_matrix, dist_coeff, essential_matrix, image_left, image_right;
string undistorted_image_list[2];
vector<Point> co_ordinates_left;
vector<Point> co_ordinates_right;
Mat R1, R2, t, mask;
double focal = 717.252;
//double focal = 1;
Point2d pp = Point2d(300, 400);
ofstream out("RT_Values.txt");
Matx33f w(0, -1, 0, 1, 0, 0, 0, 0, 1);

Mat W = Mat(w);


//This function reads the parameters obtained from caliberation. The data is read out of an XML file. 
//This function fills in the gloabal variables for intrinsic matrix and Distortion coefficients 
void readCameraParameters(char* input_filename) {
	FileStorage xml_file;
	xml_file.open(input_filename, FileStorage::READ);
	if (!xml_file.isOpened()) {
		cerr << " Fail to open " << input_filename << endl;
		exit(EXIT_FAILURE);
	}

	xml_file["camera_matrix"] >> intrinsic_matrix;
	xml_file["distortion_coefficients"] >> dist_coeff;

	//cout << "\n Intrinsic_Matrix = " << endl << " " << intrinsic_matrix << endl << endl;
	//cout << " Distortion_coefficients = " << endl << " " << dist_coeff << endl << endl;

	xml_file.release();
}

//This function catches all the clicks done on the left image and loads it onto a vector.
void click_matching_left(int event, int x, int y, int flags, void* param) {
	if (event == CV_EVENT_LBUTTONDOWN) {
		vector<Point>* points_left = (vector<Point>*)param;
		points_left->push_back(Point(x, y));
	}
}

//This function catches all the clicks done on the right image and loads it onto a vector.
void click_matching_right(int event, int x, int y, int flags, void* param) {
	vector<Point>* points_right = (vector<Point>*)param;
	if (event == CV_EVENT_LBUTTONDOWN) {
		points_right->push_back(Point(x, y));
	}
}

//This function computes the essential matrix. It calls the above click registering functions 
//and uses them in its computation. It also updates the mask variable which records inliers and outliers
//HAVE TO SELECT ATLEAST 7 POINTS OR ELSE PROGRAM FAILS
void compute_essential_matrix(string left_img, string right_img) {
	

	image_left = imread(left_img, IMREAD_COLOR);
	image_right = imread(right_img, IMREAD_COLOR);

	namedWindow("Left_Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Right_Image", CV_WINDOW_AUTOSIZE);

	imshow("Left_Image", image_left);
	imshow("Right_Image", image_right);

//Track image clicks for left and right images

	setMouseCallback("Left_Image", click_matching_left, (void*)&co_ordinates_left);
	setMouseCallback("Right_Image", click_matching_right, (void*)&co_ordinates_right);
	waitKey();

	destroyAllWindows();

	cout << "Points_left are:" << co_ordinates_left.size() << endl;
	cout << "Points_right are:" << co_ordinates_right.size() << endl;

	essential_matrix = findEssentialMat(co_ordinates_left, co_ordinates_right, focal, pp, RANSAC, 0.999, 3, mask);
	cout << "essential matrix:" << essential_matrix << endl;
	out << "Chosen points on the Left image are:" << endl << endl << co_ordinates_left << endl;
	out << endl << endl << endl;
	out << "Chosen points on the Right image are:" << endl << endl << co_ordinates_right << endl;
	out << endl << endl << endl;
	out << "Computed Essential Matrix is:" << endl << endl << essential_matrix << endl;
	out << endl << endl << endl;

	cout << "========================" << endl;

	/*for (int i = 0; i < mask.size().height; ++i) {
		for (int j = 0; j < mask.size().width; j++) {
			cout << (int)mask.at<uchar>(i, j) << endl;
		}
	}*/
	//cout << "mask starts " << mask.size() << endl;

}


//This function draws our epipolar lines. It calculates the fundamental matrix from matching points. The lines are computed 
// and plotted (Lines in 1st image are for matching points in 2nd image and vice versa.
// It also draws circle at matching points - Yellow for inliers and red for outliers

void drawEpipolarLines(Mat& image_out_right, Mat& image_out_left, Mat& image1, Mat& image2, Mat& k, Mat& E, vector<Point>& points1, vector<Point>& points2, int whichImage)
{
	ostringstream names;
	Mat Fundamental_matrix = findFundamentalMat(Mat(points1), Mat(points2), FM_7POINT);//Find F from 7 points atleast
	//Mat Fundamental_matrix = k.inv().t() * E * k.inv();
	vector<Vec3f> lines1, lines2;
	RNG rng(12345);
	//vector<Point> p1, p2;

	for (int i = 0; i < mask.size().height; ++i) {
		for (int j = 0; j < mask.size().width; j++) {
			if (((int)mask.at<uchar>(i, j)) == 1) {
				circle(image_out_left, points1[i], 3, Scalar(0, 255, 255), -1, 8);
				circle(image_out_right, points2[i], 3, Scalar(0, 255, 255), -1, 8);
			}
			else {
				circle(image_out_left, points1[i], 3, Scalar(0, 0, 255), -1, 8);
				circle(image_out_right, points2[i], 3, Scalar(0, 0, 255), -1, 8);
			}

		}
	}

	if (!image_out_left.empty() || !image_out_right.empty()) {
		namedWindow("Matching_Points_left", CV_WINDOW_AUTOSIZE);
		namedWindow("Matching_Points_right", CV_WINDOW_AUTOSIZE);
		imshow("Matching_Points_left", image_out_left);
		imshow("Matching_Points_right", image_out_right);
		waitKey();

		names << "Matching_Pts_Left.jpg";
		imwrite(names.str(), image_out_left);
		names.str("");
		names << "Matching_Pts_Right.jpg";
		imwrite(names.str(), image_out_right);
		names.str("");
	}
	destroyAllWindows();
	//correctMatches(Fundamental_matrix, points1, points2, p1, p2);//Did not work as expected.
	computeCorrespondEpilines(Mat(points1), 1, Fundamental_matrix, lines1);//Store line parameters for each point on 1st image
	computeCorrespondEpilines(Mat(points2), 2, Fundamental_matrix, lines2);//Store line parameters for each point on 2nd image
	//cout << "Fund_Mat: " << Fundamental_matrix << endl;	
	vector<Vec3f>::const_iterator it2 = lines2.begin();
	//Draw lines of 2nd image points in the first image and lines for the 1st image points on the 2nd image
	for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
	{
		Scalar color_line = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// Draw the line between first and last column
		line(image_out_right, Point(0, -(*it)[2] / (*it)[1]), Point(image2.size().width, -((*it)[2] + (*it)[0] * image2.size().width) / (*it)[1]), color_line);
		if (it2 != lines2.end()) {
			line(image_out_left, Point(0, -(*it2)[2] / (*it2)[1]), Point(image1.size().width, -((*it2)[2] + (*it2)[0] * image1.size().width) / (*it2)[1]), color_line);
			++it2;
		}
	}
	/*for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it)
	{
		
	}*/
	// Check mask which stores inliers as 1 and outliers as 0. The points are classified in the order they're selected
	//Plot points as yellow (inliers) or red (outliers)
	//accordingly
	//cout << "mask = "<< mask.size().height << mask.size().width <<endl;
	

	out << "The Inliers of left image are:" << endl << endl;
	for (int i = 0; i < mask.size().height; ++i) {
		for (int j = 0; j < mask.size().width; j++) {
			if (((int)mask.at<uchar>(i, j)) == 1) {
				out << points1[i] << endl;
				
			}
		}
	}
	out << endl << endl << endl;
	out << "The Outliers of left image are:" << endl << endl;
	for (int i = 0; i < mask.size().height; ++i) {
		for (int j = 0; j < mask.size().width; j++) {
			if (((int)mask.at<uchar>(i, j)) == 0) {
				out << points1[i] << endl;
			}
		}
	}
	
	out << endl << endl << endl;
	out << "The Inliers of Right image are:" << endl << endl;
	for (int i = 0; i < mask.size().height; ++i) {
		for (int j = 0; j < mask.size().width; j++) {
			if (((int)mask.at<uchar>(i, j)) == 1) {
				out << points2[i] << endl;
			}
		}
	}

	out << endl << endl << endl;
	out << "The Outliers of right image are:" << endl << endl;
	for (int i = 0; i < mask.size().height; ++i) {
		for (int j = 0; j < mask.size().width; j++) {
			if (((int)mask.at<uchar>(i, j)) == 0) {
				out << points2[i] << endl;
			}
		}
	}

	out << endl << endl << endl;

	if (!image_out_left.empty() || !image_out_right.empty()) {
		namedWindow("Epipolar_lines_left", CV_WINDOW_AUTOSIZE);
		namedWindow("Epipolar_lines_right", CV_WINDOW_AUTOSIZE);
		imshow("Epipolar_lines_left", image_out_left);
		imshow("Epipolar_lines_right", image_out_right);
		waitKey();

		names << "EpipolarLines_Left.jpg";
		imwrite(names.str(), image_out_left);
		names.str("");
		names << "EpipolarLines_Right.jpg";
		imwrite(names.str(), image_out_right);
		names.str("");
	}
	destroyAllWindows();
}

//This function calculates the depth. Takes in 2 possible rotation matrices and 1 translation matrix (+/- manipulation 
//done inside). Also it takes in camera pose for the left camera and the intrinsic matrix. Computes 4 possible P values
//through triangulate function. We then convert these world points to first camera reference frame using K*Pose1*Pw. The 
//function lets us see which of them have positive depths. Based on our input, corresponding rotation matrix and translation 
//vectors are returned with computed PL values.

tuple<Mat,Mat,Mat, Mat> find_depth(Mat R1, Mat R2, Mat t, Mat Pose1, Mat k) {
	cout << "R1 = " <<R1 << endl;
	cout << "R2 = " << R2 << endl;
	cout << "t = " << t << endl;
	Mat P2a = Mat::zeros(3, 4, CV_64FC1);
	Mat P2b = Mat::zeros(3, 4, CV_64FC1);
	Mat P2c = Mat::zeros(3, 4, CV_64FC1);
	Mat P2d = Mat::zeros(3, 4, CV_64FC1);
	Mat Pose2a, Pose2b, Pose2c, Pose2d;
	Pose2a.convertTo(Pose2a, CV_64FC1);
	Pose2b.convertTo(Pose2b, CV_64FC1);
	Pose2c.convertTo(Pose2c, CV_64FC1);
	Pose2d.convertTo(Pose2d, CV_64FC1);
	R1.convertTo(R1, CV_64FC1);
	R2.convertTo(R2, CV_64FC1);
	t.convertTo(t, CV_64FC1);
	k.convertTo(k, CV_64FC1);
	Pose1.convertTo(Pose1, CV_64FC1);
	int choice;


	for (int n = 0; n < 3; n++) {
		R1.col(n).copyTo(P2a.col(n));
	}
	t.col(0).copyTo(P2a.col(3));

	for (int n = 0; n < 3; n++) {
		R2.col(n).copyTo(P2b.col(n));
	}
	t.col(0).copyTo(P2b.col(3));

	t = -t;
	for (int n = 0; n < 3; n++) {
		R1.col(n).copyTo(P2c.col(n));
	}
	t.col(0).copyTo(P2c.col(3));

	for (int n = 0; n < 3; n++) {
		R2.col(n).copyTo(P2d.col(n));
	}
	t.col(0).copyTo(P2d.col(3));

	Pose2a = k*P2a;
	Pose2b = k*P2b;
	Pose2c = k*P2c;
	Pose2d = k*P2d;
	

	Mat world_pts1, world_pts2, world_pts3, world_pts4;
	Mat left_pts = Mat(co_ordinates_left);
	Mat right_pts = Mat(co_ordinates_right);

	world_pts1.convertTo(world_pts1, CV_64FC1);
	world_pts2.convertTo(world_pts2, CV_64FC1);
	world_pts3.convertTo(world_pts3, CV_64FC1);
	world_pts4.convertTo(world_pts4, CV_64FC1);
	left_pts.convertTo(left_pts, CV_64FC1);
	right_pts.convertTo(right_pts, CV_64FC1);
	
	triangulatePoints(Pose1, Pose2a, left_pts, right_pts, world_pts1);
	triangulatePoints(Pose1, Pose2b, left_pts, right_pts, world_pts2);
	triangulatePoints(Pose1, Pose2c, left_pts, right_pts, world_pts3);
	triangulatePoints(Pose1, Pose2d, left_pts, right_pts, world_pts4);
	
	Mat PLa, PLb, PLc, PLd;
	PLa = P2a * world_pts1;
	PLb = P2b * world_pts2;
	PLc = P2c * world_pts3;
	PLd = P2d * world_pts4;


	cout << "PLa = " << endl << PLa << endl;
	cout << "PLb = " << endl << PLb << endl;
	cout << "PLc = " << endl << PLc << endl;
	cout << "PLd = " << endl << PLd << endl;

	cout << "Enter the numbers 1, 2, 3, 4 depending on which positive depth matrix you want to choose" << endl;

	cin >> choice;
	 
	if (choice == 1)
		return make_tuple(R1, -t, PLa, world_pts1);
	else if (choice == 2)
		return make_tuple(R2, -t, PLb, world_pts2);
	else if (choice == 3)
		return make_tuple(R1, t, PLc, world_pts3);
	else
		return make_tuple(R2, t, PLd, world_pts4);
}

//This function calculates PR value
Mat Clc_Pr(Mat Rotation, Mat Translation, Mat wrlds_pts) {
	Mat P = Mat::zeros(3, 4, CV_64FC1);
	for (int n = 0; n < 3; n++) {
		Rotation.col(n).copyTo(P.col(n));
	}
	Translation.col(0).copyTo(P.col(3));
	
	Mat PR = P * wrlds_pts;
	return PR;

}

//Compute reprojection error
void compute_reprojection_error(vector<Point> actual, vector<Point> projected) {
	double error_x = 0, error_y = 0, total_error;
	for (int i = 0; i < actual.size(); ++i) {
		error_x += pow(projected[i].x - actual[i].x, 2);
		error_y += pow(projected[i].y - actual[i].y, 2);
	}
	total_error = error_x + error_y;
	total_error = sqrt(total_error /( 2*actual.size()));

	out << endl << endl;
	out << "Reprojection Error is: " << total_error << " " << "Pixels" << endl;

}



//This function performs SVD on Essential matrix. Then computes 2 possible R matrices as U*W*Vtrans and U*Wtrans*Vtrans
//The world reference frame is assumed to be alligned with left camera. Hence its Pose is [I 0]. The pose of the second 
//Camera is [R|T]. It calls the find depth function to let us choose the right rotation and trans matrix. We then calculate
//PR as R*PL + t. We then convert PR to pixel coordinates and plot on the images.
void compute_Rotation_translation(Mat essent_matr, Mat K) {

	ostringstream title;
	
	SVD E = SVD(essent_matr);

	Mat U = E.u;
	Mat V = E.vt;
	cout << "U = " << U << endl;

	cout << "V = " << V << endl;
	
	double det_U = determinant(U);
	double det_V = determinant(V);
	Mat project_points, project_points2, project_points3, project_points4;
	cout << "det_U = " << det_U << endl;
	
	cout << "det_V = " << det_V << endl;
	//decomposeEssentialMat(essent_matr, R1, R2, t);
	U.convertTo(U, CV_64FC1);
	V.convertTo(V, CV_64FC1);
	W.convertTo(W, CV_64FC1);
	Mat Rot1 = U*W*V;
	Mat Rot2 = U*W.t()*V;
	cout << "Rot1 = " << Rot1 << endl << "Rot2 = " << Rot2 << endl;
	Mat points_mat = Mat(co_ordinates_left);
	points_mat.convertTo(points_mat, CV_32F);
	
	vector<Point3f> P;
	Mat A = Mat::eye(3, 3, CV_64FC1);
	Mat C = Mat::zeros(3, 1, CV_64FC1);
	Mat q1 = Mat::zeros(3, 4, CV_64FC1);
	Mat q2 = Mat::zeros(3, 4, CV_64FC1);
	Mat P1, P2;
	//Mat S = Mat(R2);
	//Mat T = Mat(t);
	Mat tr = Mat::zeros(3, 1, CV_64FC1);;
	U.col(2).copyTo(tr.col(0));
	cout << "tr = " << tr << endl;
	for (int m = 0; m < 3; m++) {
		A.col(m).copyTo(q1.col(m));
	}
	C.col(0).copyTo(q1.col(3));

	P1 = K*q1;
	
	cout << "q1 = " << q1 << endl;

	tuple<Mat, Mat, Mat, Mat> RandT = find_depth(Rot1, Rot2, tr, P1, K);

	Mat R = get<0>(RandT);
	Mat T = get<1>(RandT);
	Mat PL = get<2>(RandT);
	Mat WP = get<3>(RandT);

	cout << "R FINAL ONE WILL BE " << get<0>(RandT) << endl;
	cout << "T FINAL ONE WILL BE " << get<1>(RandT) << endl;
	cout << "PL FINAL ONE WILL BE " << get<2>(RandT) << endl;

	Mat PR;
	T.convertTo(T, CV_64FC1);
	R.convertTo(R, CV_64FC1);
	PL.convertTo(PL, CV_64FC1);
	PR = Clc_Pr(R, T, WP);
	PR.convertTo(PR, CV_64FC1);
	
	Mat X;
	X = K * PR;
	X.convertTo(X, CV_64FC1);
	project_points.convertTo(project_points, CV_64FC1);
	convertPointsFromHomogeneous(X.t(), project_points);

	cout << endl << endl << "Actual right points are:" << endl << co_ordinates_right << endl;

	cout << endl << endl << "Projected points are:" << endl << project_points << endl;

	out << "Rotation Matrix is:" << endl << endl << R << endl;
	out << endl << endl << endl;
	out << "Translation Vector is:" << endl << endl << T << endl;
	out << endl << endl << endl;
	out << "PL is:" << endl << endl << PL << endl;
	out << endl << endl << endl;
	out << "Actual points on the right image were:" << endl << endl << co_ordinates_right << endl;
	out << endl << endl << endl;
	
	vector<Point> PPP = project_points;
	out << "Projected Points on the right image are:" << endl << endl << PPP << endl;
	compute_reprojection_error(co_ordinates_right, PPP);
	out.close();
	for (int i = 0; i < PPP.size(); ++i) {
		cout << PPP[i] << endl;
		circle(image_right, PPP[i], 3, Scalar(0, 255, 0), -1, 8);
	}
	namedWindow("Projected_Points", CV_WINDOW_AUTOSIZE);
	imshow("Projected_Points", image_right);
	waitKey();
	title << "Projected_Points.jpg";
	imwrite(title.str(), image_right);
	destroyAllWindows();

}

int main(int argc, char* argv[])
{

	Mat image, new_intrinsic_matrix, new_image;
	double alpha = 1;
	Size newImgSize, image_size;
	Rect* validPixROI = 0;
	bool centerPrincipalPoint;
	ostringstream name;


	if (argc < 2)
	{
		cout << "No images entered" << endl;
		return -1;
	}


	readCameraParameters("out_camera_data.xml");
	
	for (int num = 1; num < argc; num++) {
		image = imread(argv[num], IMREAD_COLOR); 

		if (image.empty())                      
		{
			cout << "Could not open/find the image" << std::endl;
			return -1;
		}

		else
		{
			//Get new intrinsic matrix, undistort the images and store them. Also, store their names as strings to open 
			//later.
			image_size = image.size();
			cout << image_size << endl;
			new_intrinsic_matrix = getOptimalNewCameraMatrix(intrinsic_matrix, dist_coeff, image_size, alpha,  newImgSize = image_size, validPixROI, centerPrincipalPoint = false);
			cout << "New_matrix is: " << new_intrinsic_matrix << endl;
			undistort(image, new_image, new_intrinsic_matrix, dist_coeff);
			name << "Undistorted_" << argv[num] << ".jpg";
			undistorted_image_list[num - 1] = name.str();
			imwrite(name.str(), new_image);
			name.str("");
		}
	}

	//Call functions to do their corresponding jobs.
	compute_essential_matrix(undistorted_image_list[0], undistorted_image_list[1]);
	drawEpipolarLines(image_right, image_left, image_left, image_right, new_intrinsic_matrix, essential_matrix, co_ordinates_left, co_ordinates_right, 1);
	compute_Rotation_translation(essential_matrix, new_intrinsic_matrix);
	
}		
