#include "cgvk.h"
#include <string.h>

int main(int argc, char** argv)
{
    cgvk_init(argv[0]);

    while (cgvk_pump_events()) {
       cgvk_render();
    }

    cgvk_quit();
    return 0;
}
