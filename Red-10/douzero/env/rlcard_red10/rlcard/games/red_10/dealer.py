# -*- coding: utf-8 -*-
''' Implement Doudizhu Dealer class
'''
import functools

from rlcard.utils import init_standard_deck
from rlcard.games.red_10.utils import cards2str, doudizhu_sort_card
from rlcard.games.base import Card

class Red_10Dealer:
    ''' Dealer will shuffle, deal cards, and determine players' roles
    '''
    def __init__(self, np_random):
        '''Give dealer the deck

        Notes:
            1. deck with 54 cards including black joker and red joker
        '''
        self.np_random = np_random
        self.deck = init_standard_deck()
        self.deck.sort(key=functools.cmp_to_key(doudizhu_sort_card))
        self.landlord = []

    def shuffle(self):
        ''' Randomly shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        ''' Deal cards to players

        Args:
            players (list): list of DoudizhuPlayer objects
        '''
        hand_num = (len(self.deck) ) // len(players)
        for index, player in enumerate(players):
            current_hand = self.deck[index*hand_num:(index+1)*hand_num]
            current_hand.sort(key=functools.cmp_to_key(doudizhu_sort_card))
            player.set_current_hand (current_hand)
            current_color = []
            for card in current_hand:
                if card.rank=="T":
                    current_color.append(card.suit)
            player.set_current_color (current_color)
            player.initial_hand = cards2str(player.current_hand)

    def determine_role(self, players):
        ''' Determine landlord and peasants according to players' hand

        Args:
            players (list): list of DoudizhuPlayer objects

        Returns:
            int: landlord's player_id
        '''
        # deal cards
        self.shuffle()
        self.deal_cards(players)
        #判定持有红10的为landloard，否则为peasant
        card10_h=Card("H","T")
        card10_d=Card("D","T")
        for player in players:
            if card10_h in player.current_hand or card10_d in player.current_hand:
                player.role="landlord"
                self.landlord.append(player)
            else:
                player.role="peasant"
        #players[0].role = 'landlord_1'
        #self.landlord = players[0]
        #players[1].role = 'peasant'
        #players[2].role = 'peasant'
        #players[0].role = 'peasant'
        #self.landlord = players[0]

        ## determine 'landlord'
        #max_score = get_landlord_score(
        #    cards2str(self.landlord.current_hand))
        #for player in players[1:]:
        #    player.role = 'peasant'
        #    score = get_landlord_score(
        #        cards2str(player.current_hand))
        #    if score > max_score:
        #        max_score = score
        #        self.landlord = player
        #self.landlord.role = 'landlord'

        # give the 'landlord' the  three cards
        return [lld.player_id for lld in self.landlord]
