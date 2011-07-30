/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence;

import org.newdawn.slick.*;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;

import towerdefence.engine.levelLoader.LevelLoader;

public class TowerDefence extends StateBasedGame
{
     static int height = 22*24;
     static int width = 29*24;

     static boolean fullscreen = false;

     static boolean showFPS = false;

     static String title = "Tower Defence";

     static int fpslimit = 60;

     public static final int LEVELSELECTSTATE = 0;
     public static final int GAMEPLAYSTATE = 1;

     

     public TowerDefence() throws SlickException
     {
          super(title);

//          this.addState(new LevelSelectState());
//          this.enterState(LEVELSELECTSTATE);
          LevelLoader level = new LevelLoader("levels/new1.xml");
          GameplayState gameplaystate = new GameplayState();
          gameplaystate.setLevel(level);
          this.addState(gameplaystate);
          this.enterState(TowerDefence.GAMEPLAYSTATE);
     }

     public static void main(String[] args) throws SlickException
     {
          AppGameContainer app = new AppGameContainer(new ScalableGame(new TowerDefence(),width,height));

          app.setDisplayMode((width), (height), fullscreen);
          app.setSmoothDeltas(true);
          app.setTargetFrameRate(fpslimit);
          app.setShowFPS(showFPS);
          app.start();
     }

    @Override
    public void initStatesList(GameContainer container) throws SlickException {
    }

    
     
}