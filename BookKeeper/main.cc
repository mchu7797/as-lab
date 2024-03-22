#include <iostream>

#include "book_keeper_ui.h"
#include "book_keeper.h"

int main() {
    book_keeper::BookKeeper bookKeeper;
    book_keeper::BookKeeperUI bookKeeperUi(&bookKeeper);

    bookKeeperUi.StartConsoleApp();

    return 0;
}
