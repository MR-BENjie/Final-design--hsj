import React from 'react';
import { BrowserRouter as Router, Redirect, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LeaderBoard from './view/LeaderBoard';
import { PvERed10DemoView,PvEDoudizhuDemoView } from './view/PvEView';
import { DoudizhuReplayView, LeducHoldemReplayView, Red10ReplayView } from './view/ReplayView';

const navbarSubtitleMap = {
    '/leaderboard': '',
    '/replay/doudizhu': 'Doudizhu',
    '/replay/leduc-holdem': "Leduc Hold'em",
    '/replay/red_10':'Red 10',
    '/pve/doudizhu-demo': 'Doudizhu PvE Demo',
};

function App() {
    // todo: add 404 page
    return (
        <Router>
            <Navbar subtitleMap={navbarSubtitleMap} />
            <div style={{ marginTop: '75px' }}>
                <Route exact path="/">
                    <Redirect to="/leaderboard?type=game&name=red_10" />
                    {/* <Redirect to="/pve/doudizhu-demo" /> */}
                </Route>
                <Route path="/leaderboard" component={LeaderBoard} />
                <Route path="/replay/red_10" component={Red10ReplayView} />
                <Route path="/pve/doudizhu-demo" component={PvEDoudizhuDemoView} />
                <Route path="/pve/red10-demo" component={PvERed10DemoView} />
            
            </div>
        </Router>
    );
}

export default App;
