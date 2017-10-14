#pragma version(1)
#pragma rs java_package_name(net.sourceforge.opencamera)
#pragma rs_fp_relaxed

#define BACKGROUND_WEIGHT_THRESHOLD (0.75f)
#define NUM_FRAMES (3)
#define NUM_DISTS (3)
#define LEARNING_RATE (0.35f)
#define MATCH_STD_ZSCORE (2.00f)

rs_allocation additional_frames[NUM_FRAMES-1];
float init_std;
bool is_first_run;

static float gaussian_dist(const float x, const float mean, const float std) {
    float two_var = 2*pow(std, 2);
    float prob_density = 1 / sqrt(M_PI * two_var) * (pow(M_E, -pow(x-mean, 2)/two_var));
    return prob_density;
}

static void next_mean_std(const float mean, const float std, const float cur_value,
                          float * next_mean, float * next_std) {
    float ro = LEARNING_RATE * gaussian_dist(cur_value, mean, std);
    *next_mean = ((1-ro) * mean + ro * cur_value);
    *next_std = sqrt((1-ro) * pow(std, 2) + ro * pow(cur_value-(*next_mean), 2));
}

static float next_dist_weight(const float weight, bool is_match) {
    if (is_match) {
        return (1-LEARNING_RATE)*weight + LEARNING_RATE;
    }
    else {
        return (1-LEARNING_RATE)*weight;
    }
}

static bool is_within_dist(const float mean, const float std, const float value, const float std_threshold) {
    float zscore = (value - mean)/std;
    if (fabs(zscore) < std_threshold) {
        return true;
    }
    else {
        return false;
    }
}

/* sort distributions according to the weights in the descending order
 */
static void sort_dist_by_weight(const float* dist_weights, size_t * dist_weight_indexes) {
    // yeah bubble sort
    bool swapped = false;
    do {
        swapped = false;
        for(size_t i=1; i<NUM_DISTS; ++i)
        {
            const size_t prev_index = i-1;
            if(dist_weights[dist_weight_indexes[prev_index]] < dist_weights[dist_weight_indexes[i]])
            {
                int temp = dist_weight_indexes[i];
                dist_weight_indexes[i] = dist_weight_indexes[prev_index];
                dist_weight_indexes[prev_index] = temp;
                swapped = true;
            }
        }
    } while(swapped);
}

static size_t match_update_dist(size_t * dist_weight_indexes, 
                       float* dist_weights, float* dist_means, float* dist_stds,
                       const float value) {
    size_t matched_index = -1;
    bool is_match = false;

    for (size_t i=0; i<NUM_DISTS; ++i) {
        size_t j = dist_weight_indexes[i];

        is_match = is_within_dist(dist_means[j], dist_stds[j], value, MATCH_STD_ZSCORE);
        float next_weight = next_dist_weight(dist_weights[j], is_match);
        dist_weights[j] = next_weight;

        if (is_match == false) {
            continue;
        }

        matched_index = j;

        float next_mean, next_std;
        next_mean_std(dist_means[j], dist_stds[j], value, &next_mean, &next_std);
        dist_means[j] = next_mean;
        dist_stds[j] = next_std;
        break; // just want to find the first match
    }

    // if no distribution contains value, 
    // replace the distribution with the least weight with the
    // distribution modelled according to the current value
    if (is_match == false) {
        size_t j = dist_weight_indexes[NUM_DISTS-1];
        dist_means[j] = value;
        dist_stds[j] = value /2;
    }

    sort_dist_by_weight(dist_weights, dist_weight_indexes);

    return matched_index;
}

static size_t forced_update_dist(size_t target_index, const float * dist_weight_indexes, 
                                 float* dist_means, float* dist_stds, const float value) {
    if (target_index != -1) {
        float next_mean, next_std;
        next_mean_std(dist_means[target_index], dist_stds[target_index], value, &next_mean, &next_std);
        dist_means[target_index] = next_mean;
        dist_stds[target_index] = next_std;
    }
    else {
        size_t j = dist_weight_indexes[NUM_DISTS-1];
        dist_means[j] = value;
        dist_stds[j] = value /2;
    }
}

static float get_background_value(
    const size_t * dist_weight_indexes, 
    const float* dist_weights,
    const float* dist_means) {

    //return dist_means[dist_weight_indexes[0]];

    float acc_weight=0;
    float average=0;
    size_t i=0;
    do {
        size_t j = dist_weight_indexes[i];
        acc_weight += dist_weights[j];
        average += dist_weights[j] * dist_means[j];
        if (acc_weight > BACKGROUND_WEIGHT_THRESHOLD) {
            break;
        }
        ++i;
    } while (i<NUM_DISTS);
    return (average / acc_weight);
}

uchar4 RS_KERNEL root(const uchar4 in, uint32_t x, uint32_t y) {
    size_t r_weight_index[NUM_DISTS];
    size_t g_weight_index[NUM_DISTS];
    size_t b_weight_index[NUM_DISTS];
    float r_weight[NUM_DISTS], g_weight[NUM_DISTS], b_weight[NUM_DISTS];
    float r_mean[NUM_DISTS], g_mean[NUM_DISTS], b_mean[NUM_DISTS];
    float r_std[NUM_DISTS], g_std[NUM_DISTS], b_std[NUM_DISTS];
    
    if (is_first_run) {
        // initialize distributions
        float initial_weight = 1.0f / NUM_DISTS;
        for (size_t i=0; i<NUM_DISTS; ++i) {
            r_weight[i] = initial_weight;
            g_weight[i] = initial_weight;
            b_weight[i] = initial_weight;
            r_weight_index[i] = i;
            g_weight_index[i] = i;
            b_weight_index[i] = i;
            r_mean[i] = 128;
            g_mean[i] = 128;
            b_mean[i] = 128;
            r_std[i] = init_std;
            g_std[i] = init_std;
            b_std[i] = init_std;
        }
    }
    else {
        float initial_weight = 1.0f / NUM_DISTS;
        for (size_t i=0; i<NUM_DISTS; ++i) {
            r_weight[i] = initial_weight*(NUM_DISTS-i);
            g_weight[i] = initial_weight*(NUM_DISTS-i);
            b_weight[i] = initial_weight*(NUM_DISTS-i);
            r_weight_index[i] = i;
            g_weight_index[i] = i;
            b_weight_index[i] = i;
            r_mean[i] = in.r;
            g_mean[i] = in.g;
            b_mean[i] = in.b;
            r_std[i] = init_std;
            g_std[i] = init_std;
            b_std[i] = init_std;
        }
    }

    /*
    size_t matched_index = match_update_dist(r_weight_index, r_weight, r_mean, r_std, in.r);
    forced_update_dist(matched_index, r_weight_index, g_mean, g_std, in.g);
    forced_update_dist(matched_index, r_weight_index, b_mean, b_std, in.b);
    // build distributions
    for (size_t i=1; i<NUM_FRAMES; ++i) {
        uchar4 pixel = rsGetElementAt_uchar4(additional_frames[i-1], x, y);
        matched_index = match_update_dist(r_weight_index, r_weight, r_mean, r_std, pixel.r);
        forced_update_dist(matched_index, r_weight_index, g_mean, g_std, pixel.g);
        forced_update_dist(matched_index, r_weight_index, b_mean, b_std, pixel.b);
    }

    uchar4 out;
    out.r = get_background_value(r_weight_index, r_weight, r_mean);
    out.g = get_background_value(r_weight_index, r_weight, g_mean);
    out.b = get_background_value(r_weight_index, r_weight, b_mean);
    */

    match_update_dist(r_weight_index, r_weight, r_mean, r_std, in.r);
    match_update_dist(g_weight_index, g_weight, g_mean, g_std, in.g);
    match_update_dist(b_weight_index, b_weight, b_mean, b_std, in.b);
    // build distributions
    for (size_t i=1; i<NUM_FRAMES; ++i) {
        uchar4 pixel = rsGetElementAt_uchar4(additional_frames[i-1], x, y);
        match_update_dist(r_weight_index, r_weight, r_mean, r_std, pixel.r);
        match_update_dist(g_weight_index, g_weight, g_mean, g_std, pixel.g);
        match_update_dist(b_weight_index, b_weight, b_mean, b_std, pixel.b);
    }

    /*
    if (x==2000 && y==1000) {
        rsDebug("r_weight of 1st distribution: ", r_weight[r_weight_index[0]]);
    }
    */

    uchar4 out;
    out.r = get_background_value(r_weight_index, r_weight, r_mean);
    out.g = get_background_value(g_weight_index, g_weight, g_mean);
    out.b = get_background_value(b_weight_index, b_weight, b_mean);

    out.a = 255;
    return out;
}
