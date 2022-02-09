#include <unistd.h>
#include <sys/types.h>

int main() {
    setreuid(0, 0);
    setregid(0, 0);
    return execle("/bin/bash", "-p", "/home/auhon/Games/Celeste/Timer/tracer/celeste_tracer_loop", "--dump", "/dev/shm/autosplitterinfo", NULL, NULL);
}
