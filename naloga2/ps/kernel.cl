
__kernel void histogramkernel(__global unsigned char *image,
                              __global unsigned int *H, int width, int height) {

  __local unsigned int buffer[256 * 3];

  int g_i = get_global_id(0);
  int g_j = get_global_id(1);

  int l_i = get_local_id(0);
  int l_j = get_local_id(1);

  // printf(" [%d %d] ", l_i, l_j);

  int l_s = get_local_size(0);

  buffer[l_i * l_s + l_j] = 0;
  buffer[l_i * l_s + l_j + 256] = 0;
  buffer[l_i * l_s + l_j + 256 * 2] = 0;

  // printf("%d ~ %d, ", H[l_off + 256 * 2], l_i * l_s + l_j);

  // Wait for the buffer to be inited in the work group
  barrier(CLK_LOCAL_MEM_FENCE);

  // We recrate the functionality of the CPU Histogram algorithm
  // Since we don't have Histogram struct, we must offset by 256
  int offset = (g_i * width + g_j) * 4;

  atomic_inc(&buffer[image[offset + 2]]);
  atomic_inc(&buffer[image[offset + 1] + 256]);
  atomic_inc(&buffer[image[offset] + 256 * 2]);

  // Sync all the threads
  barrier(CLK_LOCAL_MEM_FENCE);

  int l_off = l_i * l_s + l_j;

  atomic_add(&H[l_off], buffer[l_off]);
  atomic_add(&H[l_off + 256], buffer[l_off + 256]);
  atomic_add(&H[l_off + 256 * 2], buffer[l_off + 256 * 2]);

  // printf("%d ~ %d, ", H[l_off + 256 * 2], l_i * l_s + l_j);
}