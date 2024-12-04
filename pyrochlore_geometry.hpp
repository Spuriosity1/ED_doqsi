#pragma once

#include <vector>
#include <armadillo>


namespace pyro {


using ivec3=arma::ivec3;

const static std::vector<ivec3> FCC_pos = {
	{0,0,0},
	{0,4,4},
	{4,0,4},
	{4,4,0}
};

const static std::vector<ivec3> dual_FCC_pos = {
	{4,4,4},
	{4,0,0},
	{0,4,0},
	{0,0,4}
};

const static std::vector<ivec3> pyro_pos = {
	{1,1,1},
	{1,-1,-1},
	{-1,1,-1},
	{-1,-1,1}
};

const ivec3 plaqt[4][6] = {
    {
	{ 0,-2, 2},
	{ 2,-2, 0},
	{ 2, 0,-2},
	{ 0, 2,-2},
	{-2, 2, 0},
	{-2, 0, 2}},
    {
	{ 0, 2,-2},
	{ 2, 2, 0},
	{ 2, 0, 2},
	{ 0,-2, 2},
	{-2,-2, 0},
	{-2, 0,-2}},
    {
	{ 0,-2,-2},
	{-2,-2, 0},
	{-2, 0, 2},
	{ 0, 2, 2},
	{ 2, 2, 0},
	{ 2, 0,-2}},
    {
	{ 0, 2, 2},
	{-2, 2, 0},
	{-2, 0,-2},
	{ 0,-2,-2},
	{ 2,-2, 0},
	{ 2, 0, 2}}
};


const static std::vector<ivec3> pyro_sites = {
{1,1,1},
{1,7,7},
{7,1,7},
{7,7,1},
{1,5,5},
{1,3,3},
{7,5,3},
{7,3,5},
{5,1,5},
{5,7,3},
{3,1,3},
{3,7,5},
{5,5,1},
{5,3,7},
{3,5,7},
{3,3,1}
};




};
