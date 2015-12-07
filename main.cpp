/***
Paper : A DWT based Steganography Approach
Author : Anirudh Kumar Agrawal , Radhika Ravi, Pratik Likhar
Roll no : 11907908
**/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <map>
#include <bitset>

using namespace cv;
using namespace std;

// Function to calculate the discrete Haar transform of a given image
Mat discreteHaarWaveletTransform(Mat image , Mat keyMatrix){
	CV_Assert(image.depth() == CV_8U);
	int nRows = image.rows;
	int nCols = image.cols;
	Mat output(image.rows, image.cols, CV_16S);
	Mat img2, img3(nRows, nCols, CV_64FC1);
	image.convertTo(img2, CV_64FC1);
	//Adding the key matrix
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols; j++){
			if (keyMatrix.at<uchar>(i, j) == 1) img2.at<double>(i, j) += 0.25;
			else if (keyMatrix.at<uchar>(i, j) == 2) img2.at<double>(i, j) -= 0.5;
			else if( keyMatrix.at<uchar>(i,j) == 3) img2.at<double>(i, j) -= 0.25;
		}
	}
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols/2; j++){
			img3.at<double>(i,j) = img2.at<double>(i,2*j) + img2.at<double>(i,2*j+1);
			img3.at<double>(i, j + nCols / 2) = img2.at<double>(i, 2 * j) - img2.at<double>(i, 2 * j + 1);
		}
	}
	img2 = img3.clone();
	for (int i = 0; i < nRows/2; i++){
		for (int j = 0; j < nCols; j++){
			img3.at<double>(i, j) = img2.at<double>(2*i,  j) + img2.at<double>(2*i+1,  j);
			img3.at<double>(i + nRows / 2, j) = img2.at<double>(2 * i, j) - img2.at<double>(2 * i + 1, j);
		}
	}
	img3.convertTo(output, CV_16SC1);
	return output;
}
// Function to find the inverse of a given haarImage
Mat inversediscreteHaarWaveletTransform(Mat haarImage,Mat &keyMatrix){
	int nRows = haarImage.rows;
	int nCols = haarImage.cols;
	map<double, int> keymap;
	keymap[0.0] = 0;
	keymap[0.25] = 1;
	keymap[0.50] = 2;
	keymap[0.75] = 3;
	Mat temp(nRows,nCols,CV_32F),temp2(nRows,nCols,CV_32F),inverse(nRows, nCols, CV_8U);
	haarImage.convertTo(temp, CV_32F);
	//Inverting the Columns
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols; j++){
			temp2.at<float>(2*i, j) = (temp.at<float>(i, j) + temp.at<float>(i + nRows/2, j))/2;
			temp2.at<float>(2*i + 1, j) = (temp.at<float>(i, j) - temp.at<float>(i + nRows / 2, j))/2;
		}
	}
	temp = temp2.clone();
	//Inverting the Rows
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols / 2; j++){
			temp2.at<float>(i, 2 * j) = (temp.at<float>(i, j) + temp.at<float>(i, j + nCols / 2))/2;
			temp2.at<float>(i, 2 * j + 1 ) = (temp.at<float>(i, j) - temp.at<float>(i, j + nCols / 2))/2;
			keyMatrix.at<uchar>(i, 2 * j) = keymap[temp2.at<float>(i, 2 * j) - floor(temp2.at<float>(i, 2 * j))];
			keyMatrix.at<uchar>(i, 2 * j + 1) = keymap[temp2.at<float>(i, 2 * j+1) - floor(temp2.at<float>(i, 2 * j+1))];
			temp2.at<float>(i, 2 * j) = round(temp2.at<float>(i, 2 * j));
			temp2.at<float>(i, 2 * j + 1) = round(temp2.at<float>(i, 2 * j + 1));
		}
	}
	temp2.convertTo(inverse, CV_8UC1);
	//return temp2;
	return inverse;
}
// Embedding the secrete message in the coverImage
Mat fixedEmbedded(Mat coverImage,vector<uchar> message,Mat &keyMatrix){
	//Calculated the haar Transform
	Mat haarImage = discreteHaarWaveletTransform(coverImage,keyMatrix);
	//Generating the message 
	int nRows = coverImage.rows, nCols = coverImage.cols;
	int nBits = nRows * nCols;
	/************************************************************************* 
	*	Embedding Process Start	
	/*************************************************************************/
	//Creating a Status Bits map
	map<pair<uchar, uchar>,vector<uchar> > statusBits;
	statusBits[pair<uchar, uchar>(0, 0)] = vector<uchar>{0, 0};
	statusBits[pair<uchar, uchar>(0, 1)] = vector<uchar>{1, 0, 0};
	statusBits[pair<uchar, uchar>(0, 2)] = vector<uchar>{1, 0};
	statusBits[pair<uchar, uchar>(0, 3)] = vector<uchar>{1};
	statusBits[pair<uchar, uchar>(1, 0)] = vector<uchar>{0, 0, 0};
	statusBits[pair<uchar, uchar>(1, 1)] = vector<uchar>{0, 1};
	statusBits[pair<uchar, uchar>(1, 2)] = vector<uchar>{1, 0, 1};
	statusBits[pair<uchar, uchar>(1, 3)] = vector<uchar>{1, 1};
	statusBits[pair<uchar, uchar>(2, 0)] = vector<uchar>{0, 0};
	statusBits[pair<uchar, uchar>(2, 1)] = vector<uchar>{0, 0, 1};
	statusBits[pair<uchar, uchar>(2, 2)] = vector<uchar>{1, 0};
	statusBits[pair<uchar, uchar>(2, 3)] = vector<uchar>{1, 1, 0};
	statusBits[pair<uchar, uchar>(3, 0)] = vector<uchar>{0};
	statusBits[pair<uchar, uchar>(3, 1)] = vector<uchar>{0, 1};
	statusBits[pair<uchar, uchar>(3, 2)] = vector<uchar>{0, 1, 0};
	statusBits[pair<uchar, uchar>(3, 3)] = vector<uchar>{1, 1};
	//
	cout << "Message Embedding begin!!!!\n";
	for (int i = 0; i < nBits / 2  ; i++){
		//embedding in the LH part
		if (i < nBits / 4){
			//Embedded in the lh subband
			//printf("%d %d %d\n", nRows / 2 + i / (nCols / 2), i % (nCols / 2), haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)));
			short diff = abs(message[2*i] - message[2*i + 1]);
			pair<uchar, uchar> temp(message[2*i], message[2*i + 1]);
			//Setting the 2 LSB for LH
			if (diff == 3) {
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 3;
				// Status Bits
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~12;
				if (statusBits[temp][0]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 1<<2 ;
			}
			else if (diff == 2){
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 2;
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~1;
				//
				if (statusBits[temp][1]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) |= 1;
				else  haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) &= ~1;
				//
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~12;
				if (statusBits[temp][0]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 1<<2;
			}
			else if (diff == 1){
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 1;
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~2;
				//status[2]
				if (statusBits[temp][2]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) |= 1;
				else  haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) &= ~1;
				//status[1]
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~12;
				if (statusBits[temp][1]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 1 << 2;
				//status[0]
				if (statusBits[temp][0]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 1 << 3;
			}
			else {
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~3;
				//
				if (statusBits[temp][1]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) |= 1;
				else  haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) &= ~1;
				//status[0]
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) &= ~12;
				if (statusBits[temp][0]) haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) |= 1 << 2;
			}
			//printf("%d %d %d %d %d\n", nRows / 2 + i / (nCols / 2), i % (nCols / 2),message[2*i],message[2*i+1], haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)));
			//printf("%d %d %d\n", message[2 * i], message[2 * i + 1], haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)));
		}
		else {
			//Embedded in the hl subband which is the top right
			short diff = abs(message[2*i] - message[2*i + 1]);
			int j = i - nBits / 4;
			//printf("%d %d\n", j / (nCols / 2), nCols / 2 + j % (nCols / 2));
			pair<uchar, uchar> temp(message[2*i], message[2*i + 1]);
			if (diff == 3) {
				haarImage.at<short>(j / (nCols / 2), nCols /2 + j% (nCols / 2)) |= 3;
				// Status Bits
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~12;
				if (statusBits[temp][0]) haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 4;
			}
			else if (diff == 2){
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 2;
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~1;
				//
				if (statusBits[temp][1]) haarImage.at<short>(nRows / 2 + j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 2;
				else  haarImage.at<short>(nRows / 2 + j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~2;
				//
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~12;
				if (statusBits[temp][0]) haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 1 << 2;
			}
			else if (diff == 1){
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 1;
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~2;
				//status[2]
				if (statusBits[temp][2]) haarImage.at<short>(nRows / 2 + j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 2;
				else  haarImage.at<short>(nRows / 2 + j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~2;
				//
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~12;
				if (statusBits[temp][1]) haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 1 << 2;
				//
				if (statusBits[temp][0]) haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 1 << 3;
			}
			else {
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~3;
				//hh
				if (statusBits[temp][1]) haarImage.at<short>(nRows / 2 + j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 1 << 1;
				else  haarImage.at<short>(nRows / 2 + j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~(1 << 1); 
				//hl
				haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) &= ~12;
				if (statusBits[temp][0]) haarImage.at<short>(j / (nCols / 2), nCols / 2 + j % (nCols / 2)) |= 1 << 2;
			}
		}
	}
	
	cout << "Message Embedded\n";
	// Get the inverse image
	Mat embeddedImage = inversediscreteHaarWaveletTransform(haarImage,keyMatrix);
	/*for (int i = nRows / 2; i < nRows; i++){
		for (int j = 0; j < nCols / 2; j++){
			printf("%d %d %d %d\n", i, j, keyMatrix.at<uchar>(i, j), haarImage.at<short>(i, j));
		}
	}*/
	//Displaying all the stages
	Mat displayImage(haarImage.rows, haarImage.cols, CV_8U);
	normalize(haarImage, displayImage, 0, 255, CV_MINMAX, CV_8U);
	namedWindow("Haar Image", WINDOW_AUTOSIZE);
	imshow("Haar Image", displayImage);
	return embeddedImage;
}
// Extracting the message
vector<uchar> messageExtraction(Mat embeddedImage,Mat keyMatrix){
	vector<uchar> message;
	Mat haarImage = discreteHaarWaveletTransform(embeddedImage,keyMatrix);
	//haarImage.convertTo(haarImage, CV_16S);
	/*************************************************************************
	*	Extraction Process Start
	/*************************************************************************/
	int nRows = haarImage.rows, nCols = haarImage.cols;
	short mask2 = 3;
	short mask34 = 3 << 2;
	cout << "Message Extraction Started \n";
	//Extracting the LH component of the image
	for (int i = nRows / 2; i < nRows ; i++){
		for (int j = 0; j < nCols / 2; j++){
			//printf("%d %d %d %d\n", i, j, keyMatrix.at<uchar>(i,j),haarImage.at<short>(i, j));
			short absolute = haarImage.at<short>(i, j) & 3;
			if (absolute == 0){
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				message.push_back(bittwo * 2 + bitone);
				message.push_back(bittwo * 2 + bitone);
			}
			else if (absolute == 1){
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				if (!bitone){
					if (bittwo == 0){
						message.push_back(1);
						message.push_back(0);
					}
					else if (bittwo == 1){
						message.push_back(3);
						message.push_back(2);
					}
					else if (bittwo == 2){
						message.push_back(0);
						message.push_back(1);
					}
					else{
						message.push_back(2);
						message.push_back(3);
					}
				}
				else{
					if (bittwo == 2){
						message.push_back(1);
						message.push_back(2);
					}
					else {
						message.push_back(2);
						message.push_back(1);
					}
				}
			}
			else if (absolute == 2){
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				if (bitone){
					if (bittwo){
						message.push_back(1);
						message.push_back(3);
					}
					else{
						message.push_back(3);
						message.push_back(1);
					}
				}
				else{
					if (bittwo){
						message.push_back(0);
						message.push_back(2);
					}
					else{
						message.push_back(2);
						message.push_back(0);
					}
				}
			}
			else {
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				if (bittwo){
					message.push_back(0);
					message.push_back(3);
				}
				else{
					message.push_back(3);
					message.push_back(0);
				}
			}
		}
	}
	//Extracting the HL component of the image
	for (int i = 0; i < nRows / 2; i++){
		for (int j = nCols/2; j < nCols; j++){
			//cout << i << " " << j << "\n";
			short absolute = haarImage.at<short>(i, j) & 3;
			if (absolute == 0){
				uchar bitone = (haarImage.at<short>(i + nRows/2, j) & 2) >> 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				message.push_back(bittwo * 2 + bitone);
				message.push_back(bittwo * 2 + bitone);
			}
			else if (absolute == 1){
				uchar bitone = (haarImage.at<short>(i + nRows / 2, j) & 2) >> 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				if (!bitone){
					if (bittwo == 0){
						message.push_back(1);
						message.push_back(0);
					}
					else if (bittwo == 1){
						message.push_back(3);
						message.push_back(2);
					}
					else if (bittwo == 2){
						message.push_back(0);
						message.push_back(1);
					}
					else{
						message.push_back(2);
						message.push_back(3);
					}
				}
				else{
					if (bittwo == 2){
						message.push_back(1);
						message.push_back(2);
					}
					else {
						message.push_back(2);
						message.push_back(1);
					}
				}
			}
			else if (absolute == 2){
				uchar bitone = (haarImage.at<short>(i + nRows / 2, j) & 2) >> 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				if (bitone){
					if (bittwo){
						message.push_back(1);
						message.push_back(3);
					}
					else{
						message.push_back(3);
						message.push_back(1);
					}
				}
				else{
					if (bittwo){
						message.push_back(0);
						message.push_back(2);
					}
					else{
						message.push_back(2);
						message.push_back(0);
					}
				}
			}
			else {
				uchar bitone = (haarImage.at<short>(i + nRows / 2, j) & 2) >> 1;
				uchar bittwo = (haarImage.at<short>(i, j) & mask34) >> 2;
				if (bittwo){
					message.push_back(0);
					message.push_back(3);
				}
				else{
					message.push_back(3);
					message.push_back(0);
				}
			}
		}
	}
	cout << "Message Extraction completed\n";
	return message;
}

//embedding code for variable method
Mat varyEmbed(Mat coverImage, vector<uchar> message, Mat keyMatrix) {

	Mat haarImage = discreteHaarWaveletTransform(coverImage, keyMatrix);

	//    imwrite("peppers_haar.jpg", haarImage);
	//Generating the message
	int nRows = coverImage.rows, nCols = coverImage.cols;
	int nBits = nRows * nCols;

	/*************************************************************************
	*	Embedding Process Start
	/*************************************************************************/
	//Creating a Status Bits map
	map<pair<uchar, uchar>, vector<uchar> > statusBits;
	statusBits[pair<uchar, uchar>(0, 0)] = vector<uchar>{0, 0};
	statusBits[pair<uchar, uchar>(0, 1)] = vector<uchar>{1, 0, 0};
	statusBits[pair<uchar, uchar>(0, 2)] = vector<uchar>{1, 0};
	statusBits[pair<uchar, uchar>(0, 3)] = vector<uchar>{1};
	statusBits[pair<uchar, uchar>(1, 0)] = vector<uchar>{0, 0, 0};
	statusBits[pair<uchar, uchar>(1, 1)] = vector<uchar>{0, 1};
	statusBits[pair<uchar, uchar>(1, 2)] = vector<uchar>{1, 0, 1};
	statusBits[pair<uchar, uchar>(1, 3)] = vector<uchar>{1, 1};
	statusBits[pair<uchar, uchar>(2, 0)] = vector<uchar>{0, 0};
	statusBits[pair<uchar, uchar>(2, 1)] = vector<uchar>{0, 0, 1};
	statusBits[pair<uchar, uchar>(2, 2)] = vector<uchar>{1, 0};
	statusBits[pair<uchar, uchar>(2, 3)] = vector<uchar>{1, 1, 0};
	statusBits[pair<uchar, uchar>(3, 0)] = vector<uchar>{0};
	statusBits[pair<uchar, uchar>(3, 1)] = vector<uchar>{0, 1};
	statusBits[pair<uchar, uchar>(3, 2)] = vector<uchar>{0, 1, 0};
	statusBits[pair<uchar, uchar>(3, 3)] = vector<uchar>{1, 1};

	map<uchar, vector<uchar> > dectobin;
	dectobin[0] = vector<uchar>{0, 0};
	dectobin[1] = vector<uchar>{0, 1};
	dectobin[2] = vector<uchar>{1, 0};
	dectobin[3] = vector<uchar>{1, 1};

	//First MxN bits embedded
	for (int i = 0; i < std::min((int)message.size() / 2, nBits / 4); i++){
		//Embedded in the HH subband, with statusbits in LH and HL subbands
		//        printf("%d %d %d\n", nRows / 2 + i / (nCols / 2), i % (nCols / 2), haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)));
		short diff = abs(message[2 * i] - message[2 * i + 1]);
		pair<uchar, uchar> temp(message[2 * i], message[2 * i + 1]);

		if (diff == 3) {
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 4) + 3;
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) % 2) + (short)statusBits[temp][0];
		}
		else if (diff == 2){
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 4) + 2;
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) % 2) + (short)statusBits[temp][0];
			haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 2) + (short)statusBits[temp][1];
		}
		else if (diff == 1){
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 4) + 1;
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) % 4) + (short)((statusBits[temp][0] << 1) | statusBits[temp][1]);
			haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 2) + (short)statusBits[temp][2];
		}
		else {
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 4) + 0;
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) = haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) - (haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) % 2) + (short)statusBits[temp][0];
			haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) - (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) % 2) + (short)statusBits[temp][1];
		}
		//        printf("%d %d %d\n", message[2 * i], message[2 * i + 1], haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)));
	}
	int j = 0, k = 0;

	// After MxN bits, embedding in 2nd LSB of LH and HL subbands with some exceptions depending on available bits
	if ((message.size()) > (nBits / 2)) {
		for (int i = 0; i < nBits / 4; i++) {
			if ((haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & 3) != 1) {
				haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) = (haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) & ~(1 << 1)) | (dectobin[message[nBits / 2 + j]][k] << 1);
				if (k == 0) k = 1;
				else {
					k = 0;
					j++;
					if (j + (nBits / 2) == message.size() || j == (nBits / 4)) break;
				}
			}
		}

		if (j + (nBits / 2) != message.size() && j != nBits / 4) {
			for (int i = 0; i < nBits / 4; i++) {
				if ((haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & 3) != 3) {
					haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & ~(1 << 1)) | (dectobin[message[nBits / 2 + j]][k] << 1);
					if (k == 0) k = 1;
					else {
						k = 0;
						j++;
						if (j + (nBits / 2) == message.size() || j == nBits / 4) break;
					}
				}
				else {
					haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & ~(1 << 0)) | (dectobin[message[nBits / 2 + j]][k] << 0);
					if (k == 0) k = 1;
					else {
						k = 0;
						j++;
						if (j + (nBits / 2) == message.size() || j == nBits / 4) break;
					}
					haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & ~(1 << 1)) | (dectobin[message[nBits / 2 + j]][k] << 1);
					if (k == 0) k = 1;
					else {
						k = 0;
						j++;
						if (j + (nBits / 2) == message.size() || j == nBits / 4) break;
					}
				}
			}
		}
	}
	//    cout << j << " " << k << "\n";

	// Case 2: Embedding in 3rd LSB of LH and then, HL subbands
	if ((message.size()) > (nBits * 3 / 4)) {
		//        int j = 0, k = 0;
		for (int i = 0; i < nBits / 4; i++) {
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) = (haarImage.at<short>(nRows / 2 + i / (nCols / 2), i % (nCols / 2)) & ~(1 << 2)) | (dectobin[message[(nBits / 2) + j]][k] << 2);
			if (k == 0) k = 1;
			else {
				k = 0;
				j++;
				if (j + (nBits / 2) == message.size() || j == (nBits / 2)) break;
			}
		}

		if (j + (nBits / 2) != message.size() && j != nBits / 2) {
			for (int i = 0; i < nBits / 4; i++) {
				haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = (haarImage.at<short>(i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & ~(1 << 2)) | (dectobin[message[(nBits / 2) + j]][k] << 2);
				if (k == 0) k = 1;
				else {
					k = 0;
					j++;
					if (j + (nBits / 2) == message.size() || j == (nBits / 2)) break;
				}
			}
		}
	}
	//    cout << j << " " << k << "\n";

	// Case 3: Embedding in 3rd LSB of HH subband
	if ((message.size()) > nBits) {
		//        int j = 0, k = 0;
		for (int i = 0; i < nBits / 4; i++) {
			haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) = (haarImage.at<short>(nRows / 2 + i / (nCols / 2), nCols / 2 + i % (nCols / 2)) & ~(1 << 2)) | (dectobin[message[(nBits / 2) + j]][k] << 2);
			if (k == 0) k = 1;
			else {
				k = 0;
				j++;
				if (j + (nBits / 2) == message.size() || j == (nBits * 5 / 8)) break;
			}
		}
	}
	//    cout << j << " " << k << "\n";


	cout << "Message Embedded\n";
	// Get the inverse image
	Mat embeddedImage = inversediscreteHaarWaveletTransform(haarImage, keyMatrix);
	//Displaying all the stages
	Mat displayImage(haarImage.rows, haarImage.cols, CV_8U);
	normalize(haarImage, displayImage, 0, 255, CV_MINMAX, CV_8U);
	namedWindow("Haar Image", WINDOW_AUTOSIZE);
	imshow("Haar Image", displayImage);
	imwrite("peppers_haar.jpg", displayImage);
	return embeddedImage;
}

//Extraction code for image using variable code 
vector<uchar> varymessageExtraction(Mat embeddedImage, Mat keyMatrix, int length){
	vector<uchar> message;
	Mat haarImage = discreteHaarWaveletTransform(embeddedImage, keyMatrix);
	/*************************************************************************
	*	Extraction Process Start
	/*************************************************************************/
	int nRows = haarImage.rows, nCols = haarImage.cols;
	cout << "Message Extraction Started \n";
	//Extracting the HH component of the image
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols / 2; j++){
			//            printf("%d %d %d\n", i, j, haarImage.at<short>(i, j));
			short absolute = haarImage.at<short>(i + (nRows / 2), j + (nCols / 2)) & 3;
			if (absolute == 0){
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = haarImage.at<short>(i + nRows / 2, j) & 1;
				message.push_back(bittwo * 2 + bitone);
				message.push_back(bittwo * 2 + bitone);
				//cout << i << " " << j << " " << bittwo * 2 + bitone << "\n";
			}
			else if (absolute == 1){
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = haarImage.at<short>(i + nRows / 2, j) & 3;
				if (!bitone){
					if (bittwo == 0){
						message.push_back(1);
						message.push_back(0);
					}
					else if (bittwo == 1){
						message.push_back(3);
						message.push_back(2);
					}
					else if (bittwo == 2){
						message.push_back(0);
						message.push_back(1);
					}
					else{
						message.push_back(2);
						message.push_back(3);
					}
				}
				else{
					if (bittwo == 2){
						message.push_back(1);
						message.push_back(2);
					}
					else {
						message.push_back(2);
						message.push_back(1);
					}
				}
			}
			else if (absolute == 2){
				uchar bitone = haarImage.at<short>(i, j + nCols / 2) & 1;
				uchar bittwo = haarImage.at<short>(i + nRows / 2, j) & 1;
				if (bitone){
					if (bittwo){
						message.push_back(1);
						message.push_back(3);
					}
					else{
						message.push_back(3);
						message.push_back(1);
					}
				}
				else{
					if (bittwo){
						message.push_back(0);
						message.push_back(2);
					}
					else{
						message.push_back(2);
						message.push_back(0);
					}
				}
			}
			else {
				uchar bittwo = haarImage.at<short>(i + nRows / 2, j) & 1;
				if (bittwo){
					message.push_back(0);
					message.push_back(3);
				}
				else{
					message.push_back(3);
					message.push_back(0);
				}
			}
			if (message.size() == length) return message;
		}
	}
	//    return message;
	//    cout << " Extracting the HL component" << message.size() << "\n";

	//Extracting the LH component of the image
	bool even = false;
	uchar temp;
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols / 2; j++){
			//cout << i << " " << j << "\n";
			if ((haarImage.at<short>(i + nRows / 2, j + nCols / 2) & 3) != 1) {
				if (even) {
					temp |= (haarImage.at<short>(i + nRows / 2, j) & 2) >> 1;
					even = false;
					message.push_back(temp);
					if (message.size() == length) return message;
				}
				else {
					temp = haarImage.at<short>(i + nRows / 2, j) & 2;
					even = true;
				}
			}
		}
	}
	//    return message;


	//Extracting HL component of the image
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols / 2; j++){
			//cout << i << " " << j << "\n";
			if ((haarImage.at<short>(i + nRows / 2, j + nCols / 2) & 3) != 3) {
				if (even) {
					temp |= (haarImage.at<short>(i, j + nCols / 2) & 2) >> 1;
					even = false;
					message.push_back(temp);
				}
				else {
					temp = haarImage.at<short>(i, j + nCols / 2) & 2;
					even = true;
				}
			}
			else {
				if (even) {
					temp |= (haarImage.at<short>(i, j + nCols / 2) & 1);
					even = false;
					message.push_back(temp);
				}
				else {
					temp = (haarImage.at<short>(i, j + nCols / 2) & 1) << 1;
					even = true;
				}
				if (even) {
					temp |= (haarImage.at<short>(i, j + nCols / 2) & 2) >> 1;
					even = false;
					message.push_back(temp);
				}
				else {
					temp = haarImage.at<short>(i, j + nCols / 2) & 2;
					even = true;
				}
			}
			if (message.size() == length) return message;
		}
	}
	//    return message;

	// Case 2: Extraction of bits in 3rd LSB of LH subband
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols / 2; j++){
			//cout << i << " " << j << "\n";
			if (even) {
				temp |= (haarImage.at<short>(i + nRows / 2, j) & 4) >> 2;
				even = false;
				message.push_back(temp);
			}
			else {
				temp = (haarImage.at<short>(i + nRows / 2, j) & 4) >> 1;
				even = true;
			}
			if (message.size() == length) return message;
		}
	}
	//    return message;

	// Case 2: Extraction of bits in 3rd LSB of HL subband
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols / 2; j++){
			//cout << i << " " << j << "\n";
			if (even) {
				temp |= (haarImage.at<short>(i, j + nCols / 2) & 4) >> 2;
				even = false;
				message.push_back(temp);
			}
			else {
				temp = (haarImage.at<short>(i, j + nCols / 2) & 4) >> 1;
				even = true;
			}
			if (message.size() == length) return message;
		}
	}
	//    return message;

	// Case 2: Extraction of bits in 3rd LSB of HH subband
	for (int i = 0; i < nRows / 2; i++){
		for (int j = 0; j < nCols / 2; j++){
			//cout << i << " " << j << "\n";
			if (even) {
				temp |= (haarImage.at<short>(i + nRows / 2, j + nCols / 2) & 4) >> 2;
				even = false;
				message.push_back(temp);
			}
			else {
				temp = (haarImage.at<short>(i + nRows / 2, j + nCols / 2) & 4) >> 1;
				even = true;
			}
			if (message.size() == length) return message;
		}
	}
	//    return message;


	cout << "Message Extraction completed\n";
	return message;
}

// Calculate the difference in the extracted and orignal message
float errorExtraction(vector<uchar> original, vector<uchar> extracted){
	int l1 = original.size();
	int l2 = extracted.size();
	if (l1 != l2) return 1;
	int count = 0;
	for (int i = 0; i < l1; i++){
		if (original[i] != extracted[i]) count++;
	}
	return (float)count / (float) l1;
}

int main(int argc, char** argv){
	if (argc != 2)
	{
		cout << " Usage: ImageSteganogrphy coverImage" << endl;
		return -1;
	}
	Mat image;
	image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Original Image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original Image", image); // Show our image inside it.
	//Variables
	int nRows = image.rows, nCols = image.cols;
	int nBits = (nRows * nCols*5)/8;
	Mat keyMatrix = Mat::zeros(nRows, nCols, CV_8UC1);
	vector<uchar> message(nBits);
	//Generating the message 
	for (int i = 0; i < nBits; i++) message[i] = rand() % 4;
	
	// Embedding the message in the coverImage
	//Mat embedded = fixedEmbedded(image, message, keyMatrix);
	Mat embedded = varyEmbed(image, message, keyMatrix);
	//Displaying the embedded image
	namedWindow("Embeddeded Image", WINDOW_AUTOSIZE);
	imshow("Embeddeded Image", embedded);
	
	// Code for external attacks on the embedded image 
	// Uncomment for Gaussian noise 
	/*
	Mat noise = Mat(embedded.size(), CV_8U);
	randn(noise, Scalar(0), Scalar(2));
	embedded += noise;

	namedWindow("Embedded Image with noise", WINDOW_AUTOSIZE);
	imshow("Embedded Image with noise", embedded);
	*/
	// Uncomment for rotation testing 
	/*
	double angle = 90;
	Point2f src_center(embedded.cols / 2.0F, embedded.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;
	warpAffine(embedded, dst, rot_mat, embedded.size());

	namedWindow("Embedded Image after rotation", WINDOW_AUTOSIZE);
	imshow("Embedded Image after rotation",dst);

	rot_mat = getRotationMatrix2D(src_center, -1.0*angle, 1.0);
	warpAffine(dst , embedded, rot_mat, embedded.size());

	namedWindow("Embedded Image after rotation correction", WINDOW_AUTOSIZE);
	imshow("Embedded Image after rotation correction", embedded);
	
	*/
	//Extracting the message
	//vector<uchar> extractedMessage = messageExtraction(embedded, keyMatrix);
	vector<uchar> extractedMessage = varymessageExtraction(embedded, keyMatrix, (int) message.size());
	//Checking the message with the original message
	cout << "Original Message\n";
	for (int i = 0; i < 20; i++)cout << message[i] << " ";
	cout << "\n";
	cout << "Extracted message\n";
	for (int i = 0; i < 20; i++)cout << extractedMessage[i] << " ";
	cout << "\n";
	cout << "Error in Message Extraction : " << errorExtraction(message, extractedMessage)*100.0 << "%\n";
	// Calculate the PSNR between the two images
	cout << "The PSNR between the images is " << cv::PSNR(image, embedded) << "\n";
	waitKey(0);
	return 0;
}